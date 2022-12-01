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
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemoryAE*********************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

/**
 * @brief Output memory block containing the inference response probabilities.
 *
 */
class ResponseMemoryAE : public ResponseMemoryProbs
{
  public:
    /**
     * @brief Construct a new Response Memory Probs object
     *
     * @param count
     * @param probs
     */
    ResponseMemoryAE(size_t count, TensorObject probs);
    /**
     * @brief Construct a new Response Memory Probs object
     *
     * @param count
     * @param tensors
     */
    ResponseMemoryAE(size_t count, tensor_map_t &&tensors);
};

/****** ResponseMemoryAEInterfaceProxy*******************/
#pragma GCC visibility push(default)
/**
 * @brief Interface proxy, used to insulate python bindings
 */
struct ResponseMemoryAEInterfaceProxy
{
    /**
     * @brief Create and initialize a ResponseMemoryAE object, and return a shared pointer to the result
     *
     * @param count
     * @param probs
     * @return std::shared_ptr<ResponseMemoryAE>
     */
    static std::shared_ptr<ResponseMemoryAE> init(cudf::size_type count, pybind11::object probs);

    /**
     * @brief Get messages count in the response memory probs object
     *
     * @param self
     * @return std::size_t
     */
    static std::size_t count(ResponseMemoryAE &self);

    /**
     * @brief Get the response memory probs object
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object get_probs(ResponseMemoryAE &self);

    /**
     * @brief Set the response memory probs object
     *
     * @param self
     * @param cupy_values
     */
    static void set_probs(ResponseMemoryAE &self, pybind11::object cupy_values);
};
#pragma GCC visibility pop

/** @} */  // end of group
}  // namespace morpheus
