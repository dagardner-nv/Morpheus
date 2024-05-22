/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/export.h"
#include "morpheus/messages/meta.hpp"

#include <boost/fiber/context.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pybind11/pytypes.h>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity

#include <filesystem>  // for path
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace morpheus {

struct MORPHEUS_EXPORT RMMHolder
{
    std::unique_ptr<rmm::device_uvector<int>> int_vector;
    std::unique_ptr<rmm::device_uvector<float>> float_vector;
};

class MORPHEUS_EXPORT CudfSourceStage : public mrc::pymrc::PythonSource<RMMHolder>
{
  public:
    using base_t = mrc::pymrc::PythonSource<RMMHolder>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    CudfSourceStage(std::size_t num_messages = 65536, std::size_t num_rows = 65536);

  private:
    subscriber_fn_t build();

    std::size_t m_num_messages;
    std::size_t m_num_rows;
};

struct MORPHEUS_EXPORT CudfSourceStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<CudfSourceStage>> init(mrc::segment::Builder& builder,
                                                                       const std::string& name,
                                                                       std::size_t num_messages,
                                                                       std::size_t num_rows);
};
/** @} */  // end of group
}  // namespace morpheus
