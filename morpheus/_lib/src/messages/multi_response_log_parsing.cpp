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

#include "morpheus/messages/multi_response_log_parsing.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseLogParsingMessage****************************************/
MultiResponseLogParsingMessage::MultiResponseLogParsingMessage(
    std::shared_ptr<morpheus::MessageMeta> meta,
    size_t mess_offset,
    size_t mess_count,
    std::shared_ptr<morpheus::ResponseMemoryLogParsing> memory,
    size_t offset,
    size_t count) :
  DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count)
{}

const TensorObject MultiResponseLogParsingMessage::get_confidences() const
{
    return this->get_output("confidences");
}

void MultiResponseLogParsingMessage::set_confidences(const TensorObject &confidences)
{
    this->set_output("confidences", confidences);
}

const TensorObject MultiResponseLogParsingMessage::get_labels() const
{
    return this->get_output("labels");
}

void MultiResponseLogParsingMessage::set_labels(const TensorObject &labels)
{
    this->set_output("labels", labels);
}

/****** MultiResponseLogParsingMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
std::shared_ptr<MultiResponseLogParsingMessage> MultiResponseLogParsingMessageInterfaceProxy::init(
    std::shared_ptr<MessageMeta> meta,
    cudf::size_type mess_offset,
    cudf::size_type mess_count,
    std::shared_ptr<ResponseMemoryLogParsing> memory,
    cudf::size_type offset,
    cudf::size_type count)
{
    return std::make_shared<MultiResponseLogParsingMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
}

std::shared_ptr<morpheus::ResponseMemoryLogParsing> MultiResponseLogParsingMessageInterfaceProxy::memory(
    MultiResponseLogParsingMessage &self)
{
    DCHECK(std::dynamic_pointer_cast<morpheus::ResponseMemoryLogParsing>(self.memory) != nullptr);

    return std::static_pointer_cast<morpheus::ResponseMemoryLogParsing>(self.memory);
}

std::size_t MultiResponseLogParsingMessageInterfaceProxy::offset(MultiResponseLogParsingMessage &self)
{
    return self.offset;
}

std::size_t MultiResponseLogParsingMessageInterfaceProxy::count(MultiResponseLogParsingMessage &self)
{
    return self.count;
}

pybind11::object MultiResponseLogParsingMessageInterfaceProxy::confidences(MultiResponseLogParsingMessage &self)
{
    // Get and convert
    auto tensor = self.get_confidences();

    return CupyUtil::tensor_to_cupy(tensor);
}

pybind11::object MultiResponseLogParsingMessageInterfaceProxy::labels(MultiResponseLogParsingMessage &self)
{
    // Get and convert
    auto tensor = self.get_labels();

    return CupyUtil::tensor_to_cupy(tensor);
}
}  // namespace morpheus
