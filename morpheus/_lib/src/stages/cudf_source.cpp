/*
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

#include "morpheus/stages/cudf_source.hpp"

#include "mrc/segment/object.hpp"
#include "pymrc/node.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/objects/file_types.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/utilities/table_util.hpp"  // for filter_null_data

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <mrc/cuda/sync.hpp>
#include <mrc/segment/builder.hpp>
#include <pybind11/cast.h>  // IWYU pragma: keep
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor
#include <pybind11/pytypes.h>   // for pybind11::int_
#include <rmm/device_buffer.hpp>

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>  // for invalid_argument
#include <utility>
#include <vector>

namespace morpheus {
// Component public implementations
// ************ CudfSourceStage ************* //
CudfSourceStage::CudfSourceStage(std::size_t num_messages, std::size_t num_rows) :
  PythonSource(build()),
  m_num_messages(num_messages),
  m_num_rows(num_rows)
{}

CudfSourceStage::subscriber_fn_t CudfSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> output) {
        // Create host data
        std::vector<int> int_col;
        std::vector<float> float_col;

        for (std::size_t row = 0; row < m_num_rows; ++row)
        {
            int_col.push_back(row);
            float_col.push_back(static_cast<float>(row) * 0.1f);
        }

        // Create rmm buffers for the columns
        auto int_buffer = std::make_shared<rmm::device_uvector<int>>(int_col.size(), rmm::cuda_stream_per_thread);

        auto float_buffer = std::make_shared<rmm::device_uvector<float>>(int_col.size(), rmm::cuda_stream_per_thread);

        MRC_CHECK_CUDA(
            cudaMemcpy(int_buffer->data(), int_col.data(), sizeof(int) * int_col.size(), cudaMemcpyHostToDevice));
        MRC_CHECK_CUDA(cudaMemcpy(
            float_buffer->data(), float_col.data(), sizeof(float) * float_col.size(), cudaMemcpyHostToDevice));

        const auto start_time{std::chrono::steady_clock::now()};

        for (std::size_t message_count = 0; message_count < m_num_messages && output.is_subscribed(); ++message_count)
        {
            auto int_buffer_copy =
                rmm::device_uvector<int>(*int_buffer, int_buffer->stream(), int_buffer->memory_resource());
            auto float_buffer_copy =
                rmm::device_uvector<float>(*float_buffer, float_buffer->stream(), float_buffer->memory_resource());

            mrc::enqueue_stream_sync_event(int_buffer_copy.stream()).get();
            mrc::enqueue_stream_sync_event(float_buffer_copy.stream()).get();

            std::vector<std::unique_ptr<cudf::column>> columns;

            columns.emplace_back(std::make_unique<cudf::column>(
                std::move(int_buffer_copy), rmm::device_buffer(0, int_buffer_copy.stream()), 0));

            columns.emplace_back(std::make_unique<cudf::column>(
                std::move(float_buffer_copy), rmm::device_buffer(0, int_buffer_copy.stream()), 0));

            auto table    = std::make_unique<cudf::table>(std::move(columns));
            auto metadata = cudf::io::table_metadata();
            metadata.schema_info.emplace_back("src_ip");
            metadata.schema_info.emplace_back("data");

            auto table_w_metadata = cudf::io::table_with_metadata{std::move(table), std::move(metadata)};
            // auto meta             = MessageMeta::create_from_cpp(std::move(table_w_metadata), 0);
            output.on_next(std::move(table_w_metadata));
        }

        const auto end_time{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{end_time - start_time};
        const auto row_size = sizeof(int) + sizeof(float);
        const auto ttl_rows = m_num_messages * m_num_rows;

        std::cerr << std::fixed << std::setprecision(2) << "CudfSourceStage\tnum_messages = " << m_num_messages
                  << "\trows = " << m_num_rows << "\ttotal_rows = " << ttl_rows << "\trow_size = " << row_size
                  << " bytes\t"
                  << "time = " << elapsed_seconds.count() << " seconds\n"
                  << "Rows/s = " << ttl_rows / elapsed_seconds.count()
                  << "\tBytes/s = " << (ttl_rows * row_size) / elapsed_seconds.count() << std::endl
                  << std::endl;

        output.on_completed();
    };
}

// ************ CudfSourceStageInterfaceProxy ************ //
std::shared_ptr<mrc::segment::Object<CudfSourceStage>> CudfSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder, const std::string& name, std::size_t num_messages, std::size_t num_rows)
{
    auto stage = builder.construct_object<CudfSourceStage>(name, num_messages, num_rows);

    return stage;
}

}  // namespace morpheus
