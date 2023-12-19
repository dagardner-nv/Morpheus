#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest import mock

import pytest

from _utils.inference_worker import IW
from morpheus.stages.inference import inference_stage
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue


def test_constructor():
    queue = ProducerConsumerQueue()
    worker = inference_stage.InferenceWorker(queue)
    assert worker._inf_queue is queue

    # Call empty methods
    worker.init()
    worker.stop()


@pytest.mark.use_python
@pytest.mark.usefixtures("config")
def test_build_output_message():

    # Pylint currently fails to work with classmethod: https://github.com/pylint-dev/pylint/issues/981
    # pylint: disable=no-member

    queue = ProducerConsumerQueue()
    worker = IW(queue)

    mock_message = mock.MagicMock()
    mock_message.meta = mock.MagicMock()
    mock_message.meta.count = 20
    mock_message.mess_offset = 11
    mock_message.mess_count = 2
    mock_message.memory = mock.MagicMock()
    mock_message.memory.count = 30
    mock_message.count = 10
    mock_message.offset = 12

    response = worker.build_output_message(mock_message)
    assert response.count == 2
    assert response.mess_offset == 11
    assert response.mess_count == 2
    assert response.offset == 0

    mock_message = mock.MagicMock()
    mock_message.meta = mock.MagicMock()
    mock_message.meta.count = 20
    mock_message.mess_offset = 11
    mock_message.mess_count = 2
    mock_message.memory = mock.MagicMock()
    mock_message.memory.count = 30
    mock_message.count = 2
    mock_message.offset = 12

    response = worker.build_output_message(mock_message)
    assert response.count == 2
    assert response.mess_offset == 11
    assert response.mess_count == 2
    assert response.offset == 0
