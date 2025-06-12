# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import typing
from functools import partial

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("gliner-preprocess")
class GliNERPreprocess(GpuAndCpuMixin, ControlMessageStage):

    def __init__(self, config: Config, *, regex_col_prefix: str = "regex_matches_"):

        super().__init__(config)
        self._regex_col_prefix = regex_col_prefix

    @property
    def name(self) -> str:
        return "gliner-preprocess"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False
    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Analyze text using an entity prediction model for sensitive data detection
        """

        df = msg.payload().copy_dataframe():
        regex_columns = [col for col in df.columns if col.startswith(self._regex_col_prefix)]
        for col in regex_columns:
            # Convert regex match columns to string type
            df = df.explode(col, ignore_index=True)



        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
