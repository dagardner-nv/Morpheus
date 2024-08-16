#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import collections.abc
import time

import pytest
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import (WrappedFunctionSourceStage,
                                               source, stage)
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.type_aliases import DataFrameType


@source
def emit_n_source(*, n: int = 10) -> collections.abc.Iterator[ControlMessage]:
    for i in range(n):
        yield ControlMessage({"metadata": {"i": i}})

@source
def endless_source() -> collections.abc.Iterator[ControlMessage]:
    i = 0
    while True:
        yield ControlMessage({"metadata": {"i": i}})
        i += 1
        time.sleep(0.1)

@source
def blocking_source() -> collections.abc.Iterator[ControlMessage]:
    yield ControlMessage({"metadata": {"i": 0}})
    while True:
        time.sleep(0.1)

@stage
def error_raiser(msg: ControlMessage, *, error_on_msg_num: int) -> ControlMessage:
    msg_metadata = msg.get_metadata()
    msg_num = msg_metadata["i"]
    print(f"Received:\n{msg_num}")

    if msg_num == error_on_msg_num:
        raise RuntimeError("Error")

    return msg

@pytest.mark.parametrize("souce_stage, error_on_msg_num",
                         [(emit_n_source, 0),
                          (emit_n_source, 5),
                          (emit_n_source, 9),
                          (endless_source, 0),
                          (endless_source, 5),
                          (blocking_source, 0)])
def test_pipeline_error_shutdown(config: Config, souce_stage: WrappedFunctionSourceStage, error_on_msg_num: int):
    """
    Test to reproduce Morpheus issue #1838
    """
    pipeline = LinearPipeline(config)
    pipeline.set_source(souce_stage(config))
    pipeline.add_stage(error_raiser(config, error_on_msg_num=error_on_msg_num))

    with pytest.raises(RuntimeError):
        pipeline.run()
