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

#include "morpheus/messages/multi_post_proc_log_parsing.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiPostprocLogParsingMessage****************************************/
MultiPostprocLogParsingMessage::MultiPostprocLogParsingMessage(
    std::shared_ptr<morpheus::MessageMeta> meta,
    size_t mess_offset,
    size_t mess_count,
    std::shared_ptr<morpheus::PostprocMemoryLogParsing> memory,
    size_t offset,
    size_t count) :
  MultiInferenceMessage(meta, mess_offset, mess_count, memory, offset, count)
{}

const TensorObject MultiPostprocLogParsingMessage::get_input_ids() const
{
    return this->get_input("input_ids");
}

void MultiPostprocLogParsingMessage::set_input_ids(const TensorObject &input_ids)
{
    this->set_input("input_ids", input_ids);
}

const TensorObject MultiPostprocLogParsingMessage::get_confidences() const
{
    return this->get_input("confidences");
}

void MultiPostprocLogParsingMessage::set_confidences(const TensorObject &confidences)
{
    this->set_input("confidences", confidences);
}

const TensorObject MultiPostprocLogParsingMessage::get_labels() const
{
    return this->get_input("labels");
}

void MultiPostprocLogParsingMessage::set_labels(const TensorObject &labels)
{
    this->set_input("labels", labels);
}

const TensorObject MultiPostprocLogParsingMessage::get_seq_ids() const
{
    return this->get_input("seq_ids");
}

void MultiPostprocLogParsingMessage::set_seq_ids(const TensorObject &seq_ids)
{
    this->set_input("seq_ids", seq_ids);
}

/****** MultiPostprocLogParsingMessageInterfaceProxy *************************/
std::shared_ptr<MultiPostprocLogParsingMessage> MultiPostprocLogParsingMessageInterfaceProxy::init(
    std::shared_ptr<MessageMeta> meta,
    cudf::size_type mess_offset,
    cudf::size_type mess_count,
    std::shared_ptr<PostprocMemoryLogParsing> memory,
    cudf::size_type offset,
    cudf::size_type count)
{
    return std::make_shared<MultiPostprocLogParsingMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

std::shared_ptr<morpheus::PostprocMemoryLogParsing> MultiPostprocLogParsingMessageInterfaceProxy::memory(
    MultiPostprocLogParsingMessage &self)
{
    DCHECK(std::dynamic_pointer_cast<morpheus::PostprocMemoryLogParsing>(self.memory) != nullptr);
    return std::static_pointer_cast<morpheus::PostprocMemoryLogParsing>(self.memory);
}

std::size_t MultiPostprocLogParsingMessageInterfaceProxy::offset(MultiPostprocLogParsingMessage &self)
{
    return self.count;
}

std::size_t MultiPostprocLogParsingMessageInterfaceProxy::count(MultiPostprocLogParsingMessage &self)
{
    return self.count;
}

pybind11::object MultiPostprocLogParsingMessageInterfaceProxy::input_ids(MultiPostprocLogParsingMessage &self)
{
    // Get and convert
    auto tensor = self.get_input_ids();

    return CupyUtil::tensor_to_cupy(tensor);
}

pybind11::object MultiPostprocLogParsingMessageInterfaceProxy::confidences(MultiPostprocLogParsingMessage &self)
{
    // Get and convert
    auto tensor = self.get_confidences();

    return CupyUtil::tensor_to_cupy(tensor);
}

pybind11::object MultiPostprocLogParsingMessageInterfaceProxy::labels(MultiPostprocLogParsingMessage &self)
{
    // Get and convert
    auto tensor = self.get_labels();

    return CupyUtil::tensor_to_cupy(tensor);
}

pybind11::object MultiPostprocLogParsingMessageInterfaceProxy::seq_ids(MultiPostprocLogParsingMessage &self)
{
    // Get and convert
    auto tensor = self.get_seq_ids();

    return CupyUtil::tensor_to_cupy(tensor);
}
}  // namespace morpheus
