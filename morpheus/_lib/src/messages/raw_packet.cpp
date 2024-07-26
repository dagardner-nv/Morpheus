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

#include "morpheus/messages/raw_packet.hpp"

#include <pybind11/pytypes.h>

#include <memory>

// We're already including pybind11.h and don't need to include cast.
// For some reason IWYU also thinks we need array for the `isinsance` call.
// IWYU pragma: no_include <pybind11/cast.h>
// IWYU pragma: no_include <array>

namespace morpheus {

namespace py = pybind11;
using namespace py::literals;

/****** Component public implementations *******************/
/****** RawPacketMessage ****************************************/

uint32_t RawPacketMessage::count() const
{
    return m_num;
}

uint32_t RawPacketMessage::get_max_size() const
{
    return m_max_size;
}

uint8_t* RawPacketMessage::get_pkt_addr_list() const
{
    return static_cast<uint8_t*>(m_packet_data->data());
}

uint32_t* RawPacketMessage::get_pkt_hdr_size_list() const
{
    return static_cast<uint32_t*>(m_header_sizes->data());
}

uint32_t* RawPacketMessage::get_pkt_pld_size_list() const
{
    return static_cast<uint32_t*>(m_payload_sizes->data());
}

uint32_t RawPacketMessage::get_queue_idx() const
{
    return m_queue_idx;
}

std::shared_ptr<RawPacketMessage> RawPacketMessage::create_from_cpp(uint32_t num,
                                                                    uint32_t max_size,
                                                                    std::unique_ptr<rmm::device_buffer>&& packet_data,
                                                                    std::unique_ptr<rmm::device_buffer>&& header_sizes,
                                                                    std::unique_ptr<rmm::device_buffer>&& payload_sizes,
                                                                    uint16_t queue_idx)
{
    return std::shared_ptr<RawPacketMessage>(
        new RawPacketMessage(num, max_size, std::move(packet_data), std::move(header_sizes), std::move(payload_sizes), queue_idx));
}

RawPacketMessage::RawPacketMessage(uint32_t num_,
                                   uint32_t max_size_,
                                   std::unique_ptr<rmm::device_buffer>&& packet_data,
                                   std::unique_ptr<rmm::device_buffer>&& header_sizes,
                                   std::unique_ptr<rmm::device_buffer>&& payload_sizes,
                                   uint16_t queue_idx_) :
  m_num(num_),
  m_max_size(max_size_),
  m_packet_data(std::move(packet_data)),
  m_header_sizes(std::move(header_sizes)),
  m_payload_sizes(std::move(payload_sizes)),
  m_queue_idx(queue_idx_)
{}

}  // namespace morpheus
