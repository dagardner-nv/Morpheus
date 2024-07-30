/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "morpheus/doca/doca_convert_stage.hpp"

#include "morpheus/doca/doca_kernels.hpp"
#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"         // for DType
#include "morpheus/types.hpp"                 // for TensorIndex
#include "morpheus/utilities/matx_util.hpp"   // for MatxUtil

#include <boost/fiber/context.hpp>
#include <boost/fiber/fiber.hpp>
#include <cuda_runtime.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>  // for data_type, type_id
#include <glog/logging.h>
#include <mrc/channel/status.hpp>  // for Status
#include <mrc/cuda/common.hpp>     // for MRC_CHECK_CUDA
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>  // for device_buffer
#include <rxcpp/rx.hpp>

#include <compare>
#include <cstdint>
#include <exception>  // for exception_ptr
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace morpheus;

std::unique_ptr<RawPacketMessage> concat_packet_buffers(std::size_t ttl_packets,
                                                        std::size_t ttl_header_bytes,
                                                        std::size_t ttl_payload_sizes_bytes,
                                                        std::size_t ttl_payload_bytes,
                                                        std::vector<std::shared_ptr<RawPacketMessage>>&& packet_buffers,
                                                        rmm::cuda_stream_view stream)
{
    DCHECK(!packet_buffers.empty());

    if (packet_buffers.size() == 1)
    {
        DCHECK(packet_buffers[0].use_count() == 1);
        return std::make_unique<RawPacketMessage>(std::move(*(packet_buffers[0])));
    }

    // At this point we don't need the header offsets, header sizes or the payload offsets
    auto combined_buffer = std::make_unique<RawPacketMessage>(
        ttl_packets,
        nullptr,
        nullptr,
        std::move(std::make_unique<rmm::device_buffer>(ttl_header_bytes, stream)),
        std::move(std::make_unique<rmm::device_buffer>(ttl_payload_sizes_bytes, stream)),
        nullptr,
        std::move(std::make_unique<rmm::device_buffer>(ttl_payload_bytes, stream))
    );

    std::size_t curr_header_offset        = 0;
    std::size_t curr_payload_offset       = 0;
    std::size_t curr_payload_sizes_offset = 0;
    for (auto& packet_buffer : packet_buffers)
    {
        auto header_addr  = static_cast<uint8_t*>(combined_buffer->m_header_buffer->data()) + curr_header_offset;
        auto payload_addr = static_cast<uint8_t*>(combined_buffer->m_payload_buffer->data()) + curr_payload_offset;
        auto payload_sizes_addr =
            static_cast<uint8_t*>(combined_buffer->m_payload_sizes->data()) + curr_payload_sizes_offset;

        MRC_CHECK_CUDA(cudaMemcpyAsync(static_cast<void*>(header_addr),
                                       packet_buffer->m_header_buffer->data(),
                                       packet_buffer->m_header_buffer->size(),
                                       cudaMemcpyDeviceToDevice,
                                       stream));

        MRC_CHECK_CUDA(cudaMemcpyAsync(static_cast<void*>(payload_addr),
                                       packet_buffer->m_payload_buffer->data(),
                                       packet_buffer->m_payload_buffer->size(),
                                       cudaMemcpyDeviceToDevice,
                                       stream));

        MRC_CHECK_CUDA(cudaMemcpyAsync(static_cast<void*>(payload_sizes_addr),
                                       packet_buffer->m_payload_sizes->data(),
                                       packet_buffer->m_payload_sizes->size(),
                                       cudaMemcpyDeviceToDevice,
                                       stream));

        curr_header_offset += packet_buffer->m_header_buffer->size();
        curr_payload_offset += packet_buffer->m_payload_buffer->size();
        curr_payload_sizes_offset += packet_buffer->m_payload_sizes->size();
    }

    MRC_CHECK_CUDA(cudaStreamSynchronize(stream));

    return combined_buffer;
}

std::unique_ptr<cudf::column> make_string_col(RawPacketMessage& packet_buffer, rmm::cuda_stream_view stream)
{
    const auto packet_count = packet_buffer.count();
    auto offsets_buffer =
        morpheus::doca::sizes_to_offsets(packet_count,
                                         static_cast<uint32_t*>(packet_buffer.m_payload_sizes->data()),
                                         stream);

    const auto offset_count     = packet_count + 1;
    const auto offset_buff_size = offset_count * sizeof(int32_t);

    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                                      offset_count,
                                                      std::move(offsets_buffer),
                                                      std::move(rmm::device_buffer(0, stream)),
                                                      0);

    return cudf::make_strings_column(
        packet_count, std::move(offsets_col), std::move(*packet_buffer.m_payload_buffer), 0, {});
}

std::unique_ptr<cudf::column> make_ip_col(RawPacketMessage& packet_buffer, rmm::cuda_stream_view stream)
{
    const auto num_packets = static_cast<morpheus::TensorIndex>(packet_buffer.count());

    // cudf doesn't support uint32, need to cast to int64 remove this once
    // https://github.com/rapidsai/cudf/issues/16324 is resolved
    auto src_type     = morpheus::DType::create<uint32_t>();
    auto dst_type     = morpheus::DType(morpheus::TypeId::INT64);

    // DevMemInfo wants a shared_ptr, but we have a unique one
    auto header_buffer = std::make_shared<rmm::device_buffer>(std::move(*packet_buffer.m_header_buffer));
    auto dev_mem_info = morpheus::DevMemInfo(header_buffer, src_type, {num_packets}, {1});

    auto ip_int64_buff = morpheus::MatxUtil::cast(dev_mem_info, dst_type.type_id());

    auto src_ip_int_col = std::make_unique<cudf::column>(cudf::data_type(dst_type.cudf_type_id()),
                                                         num_packets,
                                                         std::move(*ip_int64_buff),
                                                         std::move(rmm::device_buffer(0, stream)),
                                                         0);

    return cudf::strings::integers_to_ipv4(src_ip_int_col->view());
}
} //namespace

namespace morpheus {

DocaConvertStage::DocaConvertStage(std::chrono::milliseconds max_time_delta, std::size_t buffer_channel_size) :
  base_t(base_t::op_factory_from_sub_fn(build())),
  m_max_time_delta{max_time_delta},
  m_buffer_channel{std::make_shared<mrc::BufferedChannel<std::shared_ptr<RawPacketMessage>>>(buffer_channel_size)}
{
    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    m_stream_cpp = rmm::cuda_stream_view(m_stream);
}

DocaConvertStage::~DocaConvertStage()
{
    cudaStreamDestroy(m_stream);
}

DocaConvertStage::subscribe_fn_t DocaConvertStage::build()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        auto buffer_reader_fiber = boost::fibers::fiber([this, &output]() {
            this->buffer_reader(output);
        });

        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this](sink_type_t x) {
                this->on_raw_packet_message(x);
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                m_buffer_channel->close_channel();
                buffer_reader_fiber.join();
            }));
    };
}

void DocaConvertStage::on_raw_packet_message(sink_type_t raw_msg)
{
    auto packet_count            = raw_msg->count();

    // since we are just extracting the ip4 source address as an unsigned 32bit int, this buffer size is smaller than
    // the input buffer
    const auto ip4_buff_size = packet_count * sizeof(uint32_t);
    auto ip4_buffer = std::make_unique<rmm::device_buffer>(ip4_buff_size, m_stream_cpp);

    // gather header data
    doca::gather_header(packet_count,
                        static_cast<uint8_t*>(raw_msg->m_header_buffer->data()),
                        static_cast<int32_t*>(raw_msg->m_header_offsets->data()),
                        static_cast<uint32_t*>(ip4_buffer->data()),
                        m_stream_cpp);

    cudaStreamSynchronize(m_stream_cpp);

    raw_msg->m_header_buffer.swap(ip4_buffer);

    m_buffer_channel->await_write(std::move(raw_msg));
}

void DocaConvertStage::buffer_reader(rxcpp::subscriber<source_type_t>& output)
{
    std::vector<std::shared_ptr<RawPacketMessage>> packets;

    while (!m_buffer_channel->is_channel_closed())
    {
        std::size_t ttl_packets             = 0;
        std::size_t ttl_header_bytes        = 0;
        std::size_t ttl_payload_bytes       = 0;
        std::size_t ttl_payload_sizes_bytes = 0;
        const auto poll_end                 = std::chrono::high_resolution_clock::now() + m_max_time_delta;
        while (std::chrono::high_resolution_clock::now() < poll_end && !m_buffer_channel->is_channel_closed())
        {
            std::shared_ptr<RawPacketMessage> raw_msg{nullptr};
            auto status = m_buffer_channel->await_read_until(raw_msg, poll_end);
            
            if (status == mrc::channel::Status::success)
            {
                ttl_packets += raw_msg->count();
                ttl_header_bytes += raw_msg->get_header_size();
                ttl_payload_bytes += raw_msg->get_payload_size();
                ttl_payload_sizes_bytes += raw_msg->get_sizes_size();
                packets.emplace_back(std::move(raw_msg));
            }
        }

        if (!packets.empty())
        {
            auto combined_data = concat_packet_buffers(
                ttl_packets, ttl_header_bytes, ttl_payload_sizes_bytes, ttl_payload_bytes, std::move(packets), m_stream_cpp);
            
            send_buffered_data(output, std::move(combined_data));
            packets.clear();
        }
    }
}

void DocaConvertStage::send_buffered_data(rxcpp::subscriber<source_type_t>& output,
                                          std::unique_ptr<RawPacketMessage>&& packet_buffer)
{
    auto src_ip_col  = make_ip_col(*packet_buffer, m_stream_cpp);
    auto payload_col = make_string_col(*packet_buffer, m_stream_cpp);

    std::vector<std::unique_ptr<cudf::column>> gathered_columns;
    gathered_columns.emplace_back(std::move(src_ip_col));
    gathered_columns.emplace_back(std::move(payload_col));

    auto gathered_table = std::make_unique<cudf::table>(std::move(gathered_columns));

    auto gathered_metadata = cudf::io::table_metadata();
    gathered_metadata.schema_info.emplace_back("src_ip");
    gathered_metadata.schema_info.emplace_back("data");

    auto gathered_table_w_metadata =
        cudf::io::table_with_metadata{std::move(gathered_table), std::move(gathered_metadata)};

    auto meta = MessageMeta::create_from_cpp(std::move(gathered_table_w_metadata), 0);
    output.on_next(std::move(meta));
}

std::shared_ptr<mrc::segment::Object<DocaConvertStage>> DocaConvertStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    std::string const& name,
    std::chrono::milliseconds max_time_delta,
    std::size_t buffer_channel_size)
{
    return builder.construct_object<DocaConvertStage>(name, max_time_delta, buffer_channel_size);
}

}  // namespace morpheus
