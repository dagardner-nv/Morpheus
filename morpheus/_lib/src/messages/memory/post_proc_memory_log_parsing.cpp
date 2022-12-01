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

#include "morpheus/messages/memory/post_proc_memory_log_parsing.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>  // for size_type
#include <pybind11/pytypes.h>

#include <cstddef>
#include <map>        // this->tensors is a map
#include <stdexcept>  // for runtime_error
#include <utility>    // for move, pair

namespace morpheus {
/****** Component public implementations *******************/
/****** PostprocMemoryLogParsing ****************************************/
PostprocMemoryLogParsing::PostprocMemoryLogParsing(
    std::size_t count, TensorObject confidences, TensorObject labels, TensorObject input_ids, TensorObject seq_ids) :
  InferenceMemory(count)
{
    this->tensors["confidences"] = std::move(confidences);
    this->tensors["labels"]      = std::move(labels);
    this->tensors["input_ids"]   = std::move(input_ids);
    this->tensors["seq_ids"]     = std::move(seq_ids);
}

const TensorObject &PostprocMemoryLogParsing::get_input_ids() const
{
    auto found = this->tensors.find("input_ids");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'input_ids' not found in memory");
    }

    return found->second;
}

void PostprocMemoryLogParsing::set_input_ids(TensorObject input_ids)
{
    this->tensors["input_ids"] = std::move(input_ids);
}

const TensorObject &PostprocMemoryLogParsing::get_confidences() const
{
    auto found = this->tensors.find("confidences");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'confidences' not found in memory");
    }

    return found->second;
}

void PostprocMemoryLogParsing::set_confidences(TensorObject confidences)
{
    this->tensors["confidences"] = std::move(confidences);
}

const TensorObject &PostprocMemoryLogParsing::get_labels() const
{
    auto found = this->tensors.find("labels");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'labels' not found in memory");
    }

    return found->second;
}

void PostprocMemoryLogParsing::set_labels(TensorObject labels)
{
    this->tensors["labels"] = std::move(labels);
}

const TensorObject &PostprocMemoryLogParsing::get_seq_ids() const
{
    auto found = this->tensors.find("seq_ids");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'seq_ids' not found in memory");
    }

    return found->second;
}

void PostprocMemoryLogParsing::set_seq_ids(TensorObject seq_ids)
{
    this->tensors["seq_ids"] = std::move(seq_ids);
}

/****** PostprocMemoryLogParsingInterfaceProxy *************************/
std::shared_ptr<PostprocMemoryLogParsing> PostprocMemoryLogParsingInterfaceProxy::init(cudf::size_type count,
                                                                                       pybind11::object confidences,
                                                                                       pybind11::object labels,
                                                                                       pybind11::object input_ids,
                                                                                       pybind11::object seq_ids)
{
    // Convert the cupy arrays to tensors
    return std::make_shared<PostprocMemoryLogParsing>(count,
                                                      std::move(CupyUtil::cupy_to_tensor(confidences)),
                                                      std::move(CupyUtil::cupy_to_tensor(labels)),
                                                      std::move(CupyUtil::cupy_to_tensor(input_ids)),
                                                      std::move(CupyUtil::cupy_to_tensor(seq_ids)));
}

std::size_t PostprocMemoryLogParsingInterfaceProxy::count(PostprocMemoryLogParsing &self)
{
    return self.count;
}

pybind11::object PostprocMemoryLogParsingInterfaceProxy::get_input_ids(PostprocMemoryLogParsing &self)
{
    return CupyUtil::tensor_to_cupy(self.get_input_ids());
}

void PostprocMemoryLogParsingInterfaceProxy::set_input_ids(PostprocMemoryLogParsing &self, pybind11::object cupy_values)
{
    self.set_input_ids(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object PostprocMemoryLogParsingInterfaceProxy::get_confidences(PostprocMemoryLogParsing &self)
{
    return CupyUtil::tensor_to_cupy(self.get_confidences());
}

void PostprocMemoryLogParsingInterfaceProxy::set_confidences(PostprocMemoryLogParsing &self,
                                                             pybind11::object cupy_values)
{
    return self.set_confidences(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object PostprocMemoryLogParsingInterfaceProxy::get_labels(PostprocMemoryLogParsing &self)
{
    return CupyUtil::tensor_to_cupy(self.get_labels());
}

void PostprocMemoryLogParsingInterfaceProxy::set_labels(PostprocMemoryLogParsing &self, pybind11::object cupy_values)
{
    return self.set_labels(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object PostprocMemoryLogParsingInterfaceProxy::get_seq_ids(PostprocMemoryLogParsing &self)
{
    return CupyUtil::tensor_to_cupy(self.get_seq_ids());
}

void PostprocMemoryLogParsingInterfaceProxy::set_seq_ids(PostprocMemoryLogParsing &self, pybind11::object cupy_values)
{
    return self.set_seq_ids(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
