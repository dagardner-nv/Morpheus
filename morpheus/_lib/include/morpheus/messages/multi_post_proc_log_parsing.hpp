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

#include "morpheus/messages/memory/post_proc_memory_log_parsing.hpp"
#include "morpheus/messages/meta.hpp"  // for MessageMeta
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>  // for object

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiPostprocLogParsingMessage****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * A stronger typed version of `MultiInferenceMessage` that is used for NLP workloads. Helps ensure the
 * proper inputs are set and eases debugging.
 *
 */
class MultiPostprocLogParsingMessage : public MultiInferenceMessage
{
  public:
    /**
     * @brief Construct a new Multi Inference NLP Message object
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory object
     * @param count Message count in inference memory object
     */
    MultiPostprocLogParsingMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                                   std::size_t mess_offset,
                                   std::size_t mess_count,
                                   std::shared_ptr<morpheus::PostprocMemoryLogParsing> memory,
                                   std::size_t offset,
                                   std::size_t count);

    /**
     * @brief Returns the 'input_ids' tensor, throws a `std::runtime_error` if it does not exist.
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_input_ids() const;

    /**
     * @brief Sets a tensor named 'input_ids'.
     *
     * @param input_ids
     */
    void set_input_ids(const TensorObject& input_ids);

    /**
     * @brief Returns the 'input_mask' tensor, throws a `std::runtime_error` if it does not exist.
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_confidences() const;

    /**
     * @brief Sets a tensor named 'input_mask'.
     *
     * @param input_mask
     */
    void set_confidences(const TensorObject& confidences);

    const TensorObject get_labels() const;

    /**
     * @brief Sets a tensor named 'input_mask'.
     *
     * @param input_mask
     */
    void set_labels(const TensorObject& labels);

    /**
     * @brief Returns the 'seq_ids' tensor, throws a `std::runtime_error` if it does not exist.
     *
     * @param name
     * @return const TensorObject
     */
    const TensorObject get_seq_ids() const;

    /**
     * @brief Sets a tensor named 'seq_ids'.
     *
     * @param seq_ids
     */
    void set_seq_ids(const TensorObject& seq_ids);
};

/****** MultiPostprocLogParsingMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MultiPostprocLogParsingMessageInterfaceProxy
{
    /**
     * @brief Create and initialize a MultiPostprocLogParsingMessage, and return a shared pointer to the result
     *
     * @param meta Holds a data table, in practice a cudf DataFrame, with the ability to return both Python and
     * C++ representations of the table
     * @param mess_offset Offset into the metadata batch
     * @param mess_count Messages count
     * @param memory Holds the generic tensor data in cupy arrays that will be used for inference stages
     * @param offset Message offset in inference memory object
     * @param count Message count in inference memory object
     * @return std::shared_ptr<MultiPostprocLogParsingMessage>
     */
    static std::shared_ptr<MultiPostprocLogParsingMessage> init(std::shared_ptr<MessageMeta> meta,
                                                                cudf::size_type mess_offset,
                                                                cudf::size_type mess_count,
                                                                std::shared_ptr<PostprocMemoryLogParsing> memory,
                                                                cudf::size_type offset,
                                                                cudf::size_type count);

    /**
     * @brief Get inference memory object shared pointer
     *
     * @param self
     * @return std::shared_ptr<morpheus::PostprocMemoryLogParsing>
     */
    static std::shared_ptr<morpheus::PostprocMemoryLogParsing> memory(MultiPostprocLogParsingMessage& self);

    /**
     * @brief Get message offset
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t offset(MultiPostprocLogParsingMessage& self);

    /**
     * @brief Get messages count
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t count(MultiPostprocLogParsingMessage& self);

    /**
     * @brief Get  'input_ids' tensor as a python object, throws a `std::runtime_error` if it does not exist
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object input_ids(MultiPostprocLogParsingMessage& self);

    /**
     * @brief Get 'input_mask' tensor as a python object, throws a `std::runtime_error` if it does not exist
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object confidences(MultiPostprocLogParsingMessage& self);
    static pybind11::object labels(MultiPostprocLogParsingMessage& self);

    /**
     * @brief Get 'seq_ids' tensor as a python object, throws a `std::runtime_error` if it does not exist
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object seq_ids(MultiPostprocLogParsingMessage& self);
};
#pragma GCC visibility pop
}  // namespace morpheus
