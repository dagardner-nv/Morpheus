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

#pragma once

#include <rmm/device_buffer.hpp>  // for device_buffer

#include <cstdint>
#include <memory>

namespace morpheus {

#pragma GCC visibility push(default)
/****** Component public implementations ******************/
/****** RawPacketMessage ****************************************/

/**
 * @brief Container for class holding a list of raw packets (number of packets, max size and pointers)
 *
 */
class RawPacketMessage
{
  public:
    /**
     * @brief Return number of packets in the message
     *
     * @return uint32_t
     */
    uint32_t count() const;
    
    std::size_t get_sizes_size() const;

    /**
     * @brief Get the address of the packet list
     *
     * @return uint8_t *
     */
    uint8_t* get_pkt_addr_list() const;

    /**
     * @brief Get the header size of the packet list
     *
     * @return uintptr_t *
     */
    uint32_t* get_pkt_hdr_size_list() const;

    /**
     * @brief Get the payload size of the packet list
     *
     * @return uintptr_t *
     */
    uint32_t* get_pkt_pld_size_list() const;

    /**
     * @brief Get the queue index of the packet list
     *
     * @return uint32_t
     */
    uint32_t get_queue_idx() const;

    /**
     * @brief Create RawPacketMessage cpp object from a cpp object, used internally by `create_from_cpp`
     *
     * @param data_table
     * @param index_col_count
     * @return std::shared_ptr<RawPacketMessage>
     */
    static std::shared_ptr<RawPacketMessage> create_from_cpp(uint32_t num,
                                                             std::unique_ptr<rmm::device_buffer>&& packet_data,
                                                             std::unique_ptr<rmm::device_buffer>&& header_sizes,
                                                             std::unique_ptr<rmm::device_buffer>&& payload_sizes,
                                                             uint16_t queue_idx = 0xFFFF);

  protected:
    RawPacketMessage(uint32_t num,
                     std::unique_ptr<rmm::device_buffer>&& packet_data,
                     std::unique_ptr<rmm::device_buffer>&& header_sizes,
                     std::unique_ptr<rmm::device_buffer>&& payload_sizes,
                     uint16_t queue_idx);

    uint32_t m_num;
    std::unique_ptr<rmm::device_buffer> m_packet_data;
    std::unique_ptr<rmm::device_buffer> m_header_sizes;
    std::unique_ptr<rmm::device_buffer> m_payload_sizes;
    uint16_t m_queue_idx;
};

struct RawPacketMessageProxy
{};

#pragma GCC visibility pop
}  // namespace morpheus
