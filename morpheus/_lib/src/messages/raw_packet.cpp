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

#include <glog/logging.h> // for DCHECK

#include <memory>

namespace morpheus {

/****** Component public implementations *******************/
/****** RawPacketMessage ****************************************/

uint32_t RawPacketMessage::count() const
{
    return m_num;
}

std::size_t RawPacketMessage::get_header_size() const
{
    return m_num_header_bytes;
}

std::size_t RawPacketMessage::get_payload_size() const
{
    return m_num_payload_bytes;
}

std::size_t RawPacketMessage::get_sizes_size() const
{
    return m_header_sizes->size();
}

uint8_t* RawPacketMessage::get_pkt_addr_list() const
{
    return static_cast<uint8_t*>(m_packet_buffer->data());
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
                     std::size_t num_header_bytes,
                     std::size_t num_payload_bytes,
                                                                    std::unique_ptr<rmm::device_buffer>&& packet_buffer,
                                                                    std::unique_ptr<rmm::device_buffer>&& header_sizes,
                                                                    std::unique_ptr<rmm::device_buffer>&& payload_sizes,
                                                                    uint16_t queue_idx)
{
    return std::shared_ptr<RawPacketMessage>(
        new RawPacketMessage(num, num_header_bytes, num_payload_bytes, std::move(packet_buffer), std::move(header_sizes), std::move(payload_sizes), queue_idx));
}

RawPacketMessage::RawPacketMessage(uint32_t num_,
                     std::size_t num_header_bytes,
                     std::size_t num_payload_bytes,
                                   std::unique_ptr<rmm::device_buffer>&& packet_buffer,
                                   std::unique_ptr<rmm::device_buffer>&& header_sizes,
                                   std::unique_ptr<rmm::device_buffer>&& payload_sizes,
                                   uint16_t queue_idx_) :
  m_num(num_),
  m_num_header_bytes(num_header_bytes),
  m_num_payload_bytes(num_payload_bytes),
  m_packet_buffer(std::move(packet_buffer)),
  m_header_sizes(std::move(header_sizes)),
  m_payload_sizes(std::move(payload_sizes)),
  m_queue_idx(queue_idx_)
{
    DCHECK(m_header_sizes->size() == m_payload_sizes->size());
    DCHECK(m_num * sizeof(uint32_t) == m_header_sizes->size());
    DCHECK(m_num_header_bytes + num_payload_bytes == m_packet_buffer->size());
}

}  // namespace morpheus
