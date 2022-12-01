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

#include "morpheus/messages/memory/response_memory_log_parsing.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>  // for size_t
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseLogParsingMessage****************************************/

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
class MultiResponseLogParsingMessage : public DerivedMultiMessage<MultiResponseLogParsingMessage, MultiResponseMessage>
{
  public:
    /**
     * @brief Default copy constructor
     */
    MultiResponseLogParsingMessage(const MultiResponseLogParsingMessage &other) = default;

    /**
     * Construct a new MultiResponseLogParsingMessage object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the inference response probabilites as a tensor
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     */
    MultiResponseLogParsingMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                                   size_t mess_offset,
                                   size_t mess_count,
                                   std::shared_ptr<morpheus::ResponseMemoryLogParsing> memory,
                                   size_t offset,
                                   size_t count);

    /**
     * @brief Returns the `confidences` (probabilities) output tensor
     *
     * @return const TensorObject
     */
    const TensorObject get_confidences() const;

    /**
     * @brief Update the `confidences` output tensor. Will halt on a fatal error if the `confidences` output tensor does
     * not exist.
     *
     * @param confidences
     */
    void set_confidences(const TensorObject &confidences);

    /**
     * @brief Returns the `labels` (probabilities) output tensor
     *
     * @return const TensorObject
     */
    const TensorObject get_labels() const;

    /**
     * @brief Update the `labels` output tensor. Will halt on a fatal error if the `labels` output tensor does not
     * exist.
     *
     * @param labels
     */
    void set_labels(const TensorObject &labels);
};

/****** MultiResponseLogParsingMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiResponseLogParsingMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiResponseLogParsingMessage object, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the inference response probabilites as a tensor
     * @param offset Message offset in inference memory instance
     * @param count Message count in inference memory instance
     * @return std::shared_ptr<MultiResponseLogParsingMessage>
     */
    static std::shared_ptr<MultiResponseLogParsingMessage> init(std::shared_ptr<MessageMeta> meta,
                                                                cudf::size_type mess_offset,
                                                                cudf::size_type mess_count,
                                                                std::shared_ptr<ResponseMemoryLogParsing> memory,
                                                                cudf::size_type offset,
                                                                cudf::size_type count);

    /**
     * @brief Returns a shared pointer of a response memory probs object
     *
     * @param self
     * @return std::shared_ptr<morpheus::ResponseMemoryLogParsing>
     */
    static std::shared_ptr<morpheus::ResponseMemoryLogParsing> memory(MultiResponseLogParsingMessage &self);

    /**
     * @brief Message offset in response memory probs object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t offset(MultiResponseLogParsingMessage &self);

    /**
     * @brief Messages count in response memory probs object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t count(MultiResponseLogParsingMessage &self);

    /**
     * @brief Return the `confidences` (probabilities) output tensor
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object confidences(MultiResponseLogParsingMessage &self);

    /**
     * @brief Return the `labels` (probabilities) output tensor
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object labels(MultiResponseLogParsingMessage &self);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
