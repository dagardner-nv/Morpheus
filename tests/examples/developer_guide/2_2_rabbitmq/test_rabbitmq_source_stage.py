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
from unittest import mock

import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.utils.type_aliases import DataFrameType
from utils import TEST_DIRS


@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'developer_guide/2_2_rabbitmq/rabbitmq_source_stage.py')])
@mock.patch('pika')
def test_ex2_rabbitmq_source_constructor(mock_pika: mock.MagicMock,
                                         config: Config,
                                         filter_probs_df: DataFrameType,
                                         import_mod: typing.List[types.ModuleType]):
    rabbitmq_source = import_mod[0]
    stage = rabbitmq_source.RabbitMQSourceStage(config)
    assert isinstance(stage, SingleOutputSource)
