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

#include "morpheus/messages/memory/response_memory.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemoryLogParsing*********************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * @brief Output memory block containing the inference response probabilities.
 *
 */
class ResponseMemoryLogParsing : public ResponseMemory
{
  public:
    /**
     * @brief Construct a new Response Memory Probs object
     *
     * @param count
     * @param confidences
     * @param labels
     */
    ResponseMemoryLogParsing(size_t count, TensorObject confidences, TensorObject labels);
    /**
     * @brief Construct a new Response Memory Probs object
     *
     * @param count
     * @param tensors
     */
    ResponseMemoryLogParsing(size_t count, tensor_map_t &&tensors);

    /**
     * @brief Returns the tensor named 'confidences', throws a `std::runtime_error` if it does not exist
     *
     * @return const TensorObject&
     */
    const TensorObject &get_confidences() const;

    /**
     * @brief Update the tensor named 'confidences'
     *
     * @param confidences
     */
    void set_confidences(TensorObject confidences);

    /**
     * @brief Returns the tensor named 'labels', throws a `std::runtime_error` if it does not exist
     *
     * @return const TensorObject&
     */
    const TensorObject &get_labels() const;

    /**
     * @brief Update the tensor named 'labels'
     *
     * @param labels
     */
    void set_labels(TensorObject labels);
};

/****** ResponseMemoryLogParsingInterfaceProxy*******************/
#pragma GCC visibility push(default)
/**
 * @brief Interface proxy, used to insulate python bindings
 */
struct ResponseMemoryLogParsingInterfaceProxy
{
    /**
     * @brief Create and initialize a ResponseMemoryLogParsing object, and return a shared pointer to the result
     *
     * @param count
     * @param confidences
     * @param labels
     * @return std::shared_ptr<ResponseMemoryLogParsing>
     */
    static std::shared_ptr<ResponseMemoryLogParsing> init(cudf::size_type count,
                                                          pybind11::object confidences,
                                                          pybind11::object labels);

    /**
     * @brief Get messages count in the response memory probs object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t count(ResponseMemoryLogParsing &self);

    /**
     * @brief Get the response memory confidences object
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object get_confidences(ResponseMemoryLogParsing &self);

    /**
     * @brief Set the response memory confidences object
     *
     * @param self
     * @param cupy_values
     */
    static void set_confidences(ResponseMemoryLogParsing &self, pybind11::object cupy_values);

    /**
     * @brief Get the response memory labels object
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object get_labels(ResponseMemoryLogParsing &self);

    /**
     * @brief Set the response memory labels object
     *
     * @param self
     * @param cupy_values
     */
    static void set_labels(ResponseMemoryLogParsing &self, pybind11::object cupy_values);
};
#pragma GCC visibility pop

/** @} */  // end of group
}  // namespace morpheus
