/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/memory/response_memory_probs.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/messages/multi_response_probs.hpp"
#include "morpheus/messages/user_meta.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>  // for size_t
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseAEMessage****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * A stronger typed version of `MultiResponseMessage` that is used for inference workloads that return a probability
 * array. Helps ensure the proper outputs are set and eases debugging
 *
 */
#pragma GCC visibility push(default)
class MultiResponseAEMessage : public DerivedMultiMessage<MultiResponseAEMessage, MultiResponseProbsMessage>
{
  public:
    /**
     * @brief Default copy constructor
     */
    MultiResponseAEMessage(const MultiResponseAEMessage &other) = default;

    /**
     * Construct a new Multi Response Probs Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the inference response probabilites as a tensor
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     */
    MultiResponseAEMessage(std::shared_ptr<morpheus::UserMessageMeta> meta,
                           size_t mess_offset,
                           size_t mess_count,
                           std::shared_ptr<morpheus::ResponseMemoryProbs> memory,
                           size_t offset,
                           size_t count);

    MultiResponseAEMessage(std::shared_ptr<morpheus::UserMessageMeta> meta,
                           size_t mess_offset,
                           size_t mess_count,
                           std::shared_ptr<morpheus::ResponseMemoryProbs> memory,
                           size_t offset,
                           size_t count,
                           std::string user_id);

    std::string user_id;
};

/****** MultiResponseAEMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiResponseAEMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiResponseAEMessage object, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the inference response probabilites as a tensor
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @return std::shared_ptr<MultiResponseAEMessage>
     */
    static std::shared_ptr<MultiResponseAEMessage> init(std::shared_ptr<UserMessageMeta> meta,
                                                        cudf::size_type mess_offset,
                                                        cudf::size_type mess_count,
                                                        std::shared_ptr<ResponseMemoryProbs> memory,
                                                        cudf::size_type offset,
                                                        cudf::size_type count,
                                                        std::string user_id);

    /**
     * @brief Returns a shared pointer of a response memory probs object
     *
     * @param self
     * @return std::shared_ptr<morpheus::ResponseMemoryProbs>
     */
    static std::shared_ptr<morpheus::ResponseMemoryProbs> memory(MultiResponseAEMessage &self);

    /**
     * @brief Message offset in response memory probs object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t offset(MultiResponseAEMessage &self);

    /**
     * @brief Messages count in response memory probs object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t count(MultiResponseAEMessage &self);

    /**
     * @brief Return the `probs` (probabilities) output tensor
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object probs(MultiResponseAEMessage &self);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
