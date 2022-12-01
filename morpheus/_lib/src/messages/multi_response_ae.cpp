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

#include "morpheus/messages/multi_response_ae.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>
#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiResponseAEMessage****************************************/
MultiResponseAEMessage::MultiResponseAEMessage(std::shared_ptr<morpheus::UserMessageMeta> meta,
                                               size_t mess_offset,
                                               size_t mess_count,
                                               std::shared_ptr<morpheus::ResponseMemoryProbs> memory,
                                               size_t offset,
                                               size_t count) :
  DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count)
{
    user_id = meta->get_user_id();
}

MultiResponseAEMessage::MultiResponseAEMessage(std::shared_ptr<morpheus::UserMessageMeta> meta,
                                               size_t mess_offset,
                                               size_t mess_count,
                                               std::shared_ptr<morpheus::ResponseMemoryProbs> memory,
                                               size_t offset,
                                               size_t count,
                                               std::string user_id) :
  DerivedMultiMessage(meta, mess_offset, mess_count, memory, offset, count)
{
    this->user_id = user_id;
}

/****** MultiResponseAEMessageInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
std::shared_ptr<MultiResponseAEMessage> MultiResponseAEMessageInterfaceProxy::init(
    std::shared_ptr<UserMessageMeta> meta,
    cudf::size_type mess_offset,
    cudf::size_type mess_count,
    std::shared_ptr<ResponseMemoryProbs> memory,
    cudf::size_type offset,
    cudf::size_type count,
    std::string user_id)
{
    if (user_id == "")
    {
        user_id = meta->get_user_id();
    }

    return std::make_shared<MultiResponseAEMessage>(
        std::move(meta), mess_offset, mess_count, std::move(memory), offset, count, user_id);
}

std::shared_ptr<morpheus::ResponseMemoryProbs> MultiResponseAEMessageInterfaceProxy::memory(
    MultiResponseAEMessage &self)
{
    DCHECK(std::dynamic_pointer_cast<morpheus::ResponseMemoryProbs>(self.memory) != nullptr);

    return std::static_pointer_cast<morpheus::ResponseMemoryProbs>(self.memory);
}

std::size_t MultiResponseAEMessageInterfaceProxy::offset(MultiResponseAEMessage &self)
{
    return self.offset;
}

std::size_t MultiResponseAEMessageInterfaceProxy::count(MultiResponseAEMessage &self)
{
    return self.count;
}

pybind11::object MultiResponseAEMessageInterfaceProxy::probs(MultiResponseAEMessage &self)
{
    // Get and convert
    auto tensor = self.get_probs();

    return CupyUtil::tensor_to_cupy(tensor);
}
}  // namespace morpheus
