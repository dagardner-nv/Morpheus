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

#include "morpheus/messages/memory/response_memory_log_parsing.hpp"

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
/****** ResponseMemoryLogParsing****************************************/
ResponseMemoryLogParsing::ResponseMemoryLogParsing(size_t count, TensorObject confidences, TensorObject labels) :
  ResponseMemory(count)
{
    this->tensors["confidences"] = std::move(confidences);
    this->tensors["labels"]      = std::move(labels);
}

ResponseMemoryLogParsing::ResponseMemoryLogParsing(size_t count, tensor_map_t &&tensors) :
  ResponseMemory(count, std::move(tensors))
{
    CHECK(has_tensor("confidences")) << "Tensor: 'confidences' not found in memory";
    CHECK(has_tensor("labels")) << "Tensor: 'labels' not found in memory";
}

const TensorObject &ResponseMemoryLogParsing::get_confidences() const
{
    auto found = this->tensors.find("confidences");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'confidences' not found in memory");
    }

    return found->second;
}

void ResponseMemoryLogParsing::set_confidences(TensorObject confidences)
{
    this->tensors["confidences"] = std::move(confidences);
}

const TensorObject &ResponseMemoryLogParsing::get_labels() const
{
    auto found = this->tensors.find("labels");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'labels' not found in memory");
    }

    return found->second;
}

void ResponseMemoryLogParsing::set_labels(TensorObject labels)
{
    this->tensors["labels"] = std::move(labels);
}

/****** ResponseMemoryLogParsingInterfaceProxy *************************/
std::shared_ptr<ResponseMemoryLogParsing> ResponseMemoryLogParsingInterfaceProxy::init(cudf::size_type count,
                                                                                       pybind11::object confidences,
                                                                                       pybind11::object labels)
{
    // Conver the cupy arrays to tensors
    return std::make_shared<ResponseMemoryLogParsing>(
        count, std::move(CupyUtil::cupy_to_tensor(confidences)), std::move(CupyUtil::cupy_to_tensor(labels)));
}

std::size_t ResponseMemoryLogParsingInterfaceProxy::count(ResponseMemoryLogParsing &self)
{
    return self.count;
}

pybind11::object ResponseMemoryLogParsingInterfaceProxy::get_confidences(ResponseMemoryLogParsing &self)
{
    return CupyUtil::tensor_to_cupy(self.get_confidences());
}

void ResponseMemoryLogParsingInterfaceProxy::set_confidences(ResponseMemoryLogParsing &self,
                                                             pybind11::object cupy_values)
{
    self.set_confidences(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object ResponseMemoryLogParsingInterfaceProxy::get_labels(ResponseMemoryLogParsing &self)
{
    return CupyUtil::tensor_to_cupy(self.get_labels());
}

void ResponseMemoryLogParsingInterfaceProxy::set_labels(ResponseMemoryLogParsing &self, pybind11::object cupy_values)
{
    self.set_labels(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
