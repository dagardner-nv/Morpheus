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

#include "morpheus/messages/memory/response_memory_ae.hpp"

#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <map>  // this->tensors is a map
#include <memory>
#include <stdexcept>  // for runtime_error
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemoryAE****************************************/
ResponseMemoryAE::ResponseMemoryAE(size_t count, TensorObject probs) : ResponseMemoryProbs(count, std::move(probs)) {}

ResponseMemoryAE::ResponseMemoryAE(size_t count, tensor_map_t &&tensors) :
  ResponseMemoryProbs(count, std::move(tensors))
{}

/****** ResponseMemoryAEInterfaceProxy *************************/
std::shared_ptr<ResponseMemoryAE> ResponseMemoryAEInterfaceProxy::init(cudf::size_type count, pybind11::object probs)
{
    // Conver the cupy arrays to tensors
    return std::make_shared<ResponseMemoryAE>(count, std::move(CupyUtil::cupy_to_tensor(probs)));
}

std::size_t ResponseMemoryAEInterfaceProxy::count(ResponseMemoryAE &self)
{
    return self.count;
}

pybind11::object ResponseMemoryAEInterfaceProxy::get_probs(ResponseMemoryAE &self)
{
    return CupyUtil::tensor_to_cupy(self.get_probs());
}

void ResponseMemoryAEInterfaceProxy::set_probs(ResponseMemoryAE &self, pybind11::object cupy_values)
{
    self.set_probs(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
