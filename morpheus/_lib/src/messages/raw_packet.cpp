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
    return m_header_buffer->size();
}

std::size_t RawPacketMessage::get_payload_size() const
{
    return m_payload_buffer->size();
}

std::size_t RawPacketMessage::get_sizes_size() const
{
    return m_header_sizes->size();
}

uint32_t RawPacketMessage::get_queue_idx() const
{
    return m_queue_idx;
}

std::shared_ptr<RawPacketMessage> RawPacketMessage::create_from_cpp(uint32_t num,
                                                             std::unique_ptr<rmm::device_buffer>&& header_sizes,
                                                             std::unique_ptr<rmm::device_buffer>&& header_offsets,
                                                             std::unique_ptr<rmm::device_buffer>&& header_buffer,
                                                             std::unique_ptr<rmm::device_buffer>&& payload_sizes,
                                                             std::unique_ptr<rmm::device_buffer>&& payload_offsets,
                                                             std::unique_ptr<rmm::device_buffer>&& payload_buffer,
                                                                    uint16_t queue_idx)
{
    return std::shared_ptr<RawPacketMessage>(
        new RawPacketMessage(num, 
                             std::move(header_sizes),
                             std::move(header_offsets),
                             std::move(header_buffer),
                             std::move(payload_sizes),
                             std::move(payload_offsets),
                             std::move(payload_buffer),
                             queue_idx));
}

RawPacketMessage::RawPacketMessage(uint32_t num_,
                                                             std::unique_ptr<rmm::device_buffer>&& header_sizes,
                                                             std::unique_ptr<rmm::device_buffer>&& header_offsets,
                                                             std::unique_ptr<rmm::device_buffer>&& header_buffer,
                                                             std::unique_ptr<rmm::device_buffer>&& payload_sizes,
                                                             std::unique_ptr<rmm::device_buffer>&& payload_offsets,
                                                             std::unique_ptr<rmm::device_buffer>&& payload_buffer,
                                   uint16_t queue_idx_) :
  m_num(num_),
  m_header_sizes(std::move(header_sizes)),
  m_header_offsets(std::move(header_offsets)),
  m_header_buffer(std::move(header_buffer)),
  m_payload_sizes(std::move(payload_sizes)),
  m_payload_offsets(std::move(payload_offsets)),
  m_ppayload_buffer(std::move(payload_buffer)),
  m_queue_idx(queue_idx_)
{
    DCHECK(m_header_sizes->size() == m_payload_sizes->size());
    DCHECK(m_num * sizeof(uint32_t) == m_header_sizes->size());
}

}  // namespace morpheus
