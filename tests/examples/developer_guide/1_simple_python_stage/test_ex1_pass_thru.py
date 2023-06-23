# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import types
import typing

import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.utils.type_aliases import DataFrameType
from utils import TEST_DIRS


@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'developer_guide/1_simple_python_stage/pass_thru.py')])
def test_pass_thru_ex1(config: Config, filter_probs_df: DataFrameType, import_mod: typing.List[types.ModuleType]):
    pass_thru = import_mod[0]
    stage = pass_thru.PassThruStage(config)

    meta = MessageMeta(filter_probs_df)
    mm = MultiMessage(meta=meta)

    assert stage.on_data(mm) is mm
