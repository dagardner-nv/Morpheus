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

#include "morpheus/messages/user_meta.hpp"

#include "morpheus/objects/python_data_table.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/utilities/table_util.hpp"

#include <cudf/io/types.hpp>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {

std::shared_ptr<UserMessageMeta> UserMessageMeta::create_from_python(pybind11::object &&data_table,
                                                                     const std::string &user_id)
{
    auto data = std::make_unique<PyDataTable>(std::move(data_table));

    return std::shared_ptr<UserMessageMeta>(new UserMessageMeta(std::move(data), user_id));
}

std::shared_ptr<UserMessageMeta> UserMessageMeta::create_from_cpp(cudf::io::table_with_metadata &&data_table,
                                                                  const std::string &user_id,
                                                                  int index_col_count)
{
    // Convert to py first
    pybind11::object py_dt = cpp_to_py(std::move(data_table), index_col_count);

    auto data = std::make_unique<PyDataTable>(std::move(py_dt));

    return std::shared_ptr<UserMessageMeta>(new UserMessageMeta(std::move(data), user_id));
}

UserMessageMeta::UserMessageMeta(std::shared_ptr<IDataTable> data, const std::string user_id) :
  MessageMeta(std::move(data)),
  m_user_id(user_id)
{}

std::string UserMessageMeta::get_user_id() const
{
    return m_user_id;
}

/********** UserMessageMetaInterfaceProxy **********/
std::shared_ptr<UserMessageMeta> UserMessageMetaInterfaceProxy::init_python(pybind11::object &&data_frame,
                                                                            const std::string &user_id)
{
    return UserMessageMeta::create_from_python(std::move(data_frame), user_id);
}

cudf::size_type UserMessageMetaInterfaceProxy::count(UserMessageMeta &self)
{
    return self.count();
}

std::string UserMessageMetaInterfaceProxy::user_id(UserMessageMeta &self)
{
    return self.get_user_id();
}

pybind11::object UserMessageMetaInterfaceProxy::get_data_frame(UserMessageMeta &self)
{
    return self.get_py_table();
}

std::shared_ptr<UserMessageMeta> UserMessageMetaInterfaceProxy::init_cpp(const std::string &filename,
                                                                         const std::string &user_id)
{
    // Load the file
    auto df_with_meta = CuDFTableUtil::load_table(filename);

    return UserMessageMeta::create_from_cpp(std::move(df_with_meta), user_id);
}
}  // namespace morpheus
