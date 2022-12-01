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

#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>
#include <glog/logging.h>  // for DCHECK_NOTNULL
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>  // for size_t
#include <memory>
#include <string>
#include <utility>  // for pair
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiMessage****************************************/

/**
 * @addtogroup messages
 * @{
 * @file
 */

#pragma GCC visibility push(default)

class UserMessageMeta : public MessageMeta
{
  public:
    std::string get_user_id() const;
    static std::shared_ptr<UserMessageMeta> create_from_python(pybind11::object&& data_table,
                                                               const std::string& user_id);
    static std::shared_ptr<UserMessageMeta> create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                            const std::string& user_id,
                                                            int index_col_count = 0);

  protected:
    UserMessageMeta(std::shared_ptr<IDataTable> data, const std::string user_id);

  private:
    std::string m_user_id;
};

struct UserMessageMetaInterfaceProxy
{
    /**
     * @brief Initialize MessageMeta cpp object with the given filename
     *
     * @param filename : Filename for loading the data on to MessageMeta
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<UserMessageMeta> init_cpp(const std::string& filename, const std::string& user_id);

    /**
     * @brief Initialize MessageMeta cpp object with a given dataframe and returns shared pointer as the result
     *
     * @param data_frame : Dataframe that contains the data
     * @return std::shared_ptr<MessageMeta>
     */
    static std::shared_ptr<UserMessageMeta> init_python(pybind11::object&& data_frame, const std::string& user_id);

    /**
     * @brief Get messages count
     *
     * @param self
     * @return cudf::size_type
     */
    static cudf::size_type count(UserMessageMeta& self);

    static std::string user_id(UserMessageMeta& self);

    /**
     * @brief Get the data frame object
     *
     * @param self
     * @return pybind11::object
     */
    static pybind11::object get_data_frame(UserMessageMeta& self);
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
