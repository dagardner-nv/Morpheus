/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/types.hpp"  // for TensorSize

#include <glog/logging.h>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// TODO: move this to the morpheus::doca namespace
uint32_t const PACKETS_PER_THREAD   = 16;
uint32_t const THREADS_PER_BLOCK    = 512;
uint32_t const PACKETS_PER_BLOCK    = PACKETS_PER_THREAD * THREADS_PER_BLOCK;
uint32_t const PACKET_RX_TIMEOUT_NS = 1000000;  // 1ms //500us

uint32_t const MAX_PKT_RECEIVE = PACKETS_PER_BLOCK;
uint32_t const MAX_PKT_CONVERT = MAX_PKT_RECEIVE * 5;
uint32_t const MAX_PKT_SIZE    = 2048;
uint32_t const MAX_PKT_HDR     = 64;
uint32_t const MAX_PKT_NUM     = 65536;
uint32_t const MAX_QUEUE       = 2;
uint32_t const MAX_SEM_X_QUEUE = 4096;

uint32_t const IP_ADDR_STRING_LEN = 15;

enum doca_traffic_type
{
    DOCA_TRAFFIC_TYPE_UDP = 0,
    DOCA_TRAFFIC_TYPE_TCP = 1,
};

struct packets_info
{
    uintptr_t* pkt_addr;
    uint32_t* pkt_hdr_size;
    uint32_t* pkt_pld_size;
    int32_t packet_count_out;
    int32_t payload_size_total_out;

    char* payload_buffer_out;
    int32_t* payload_sizes_out;

    int64_t* src_mac_out;
    int64_t* dst_mac_out;
    int64_t* src_ip_out;
    int64_t* dst_ip_out;
    uint16_t* src_port_out;
    uint16_t* dst_port_out;
    int32_t* tcp_flags_out;
    int32_t* ether_type_out;
    int32_t* next_proto_id_out;
    uint32_t* timestamp_out;
};

// TODO: move this to it's own header
struct packet_data_buffer
{
    rmm::device_buffer buffer;
    morpheus::TensorSize cur_offset_bytes;
    morpheus::TensorSize elements;

    morpheus::TensorSize capacity() const
    {
        return buffer.size();
    };

    morpheus::TensorSize available_bytes() const
    {
        return capacity() - cur_offset_bytes;
    };

    bool empty() const
    {
        return cur_offset_bytes == 0;
    }

    void advance(morpheus::TensorSize num_bytes, morpheus::TensorSize num_elements)
    {
        cur_offset_bytes += num_bytes;
        elements += num_elements;
        CHECK(cur_offset_bytes <= capacity());
    }

    template <typename T = uint8_t>
    T* data()
    {
        return static_cast<T*>(buffer.data());
    }

    template <typename T = uint8_t>
    T* current_location()
    {
        return data<T>() + cur_offset_bytes;
    }

    void shrink_to_fit()
    {
        if (available_bytes() > 0)
        {
            buffer.resize(cur_offset_bytes, buffer.stream());
            buffer.shrink_to_fit(buffer.stream());
        }
    }
};
