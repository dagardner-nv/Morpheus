# Copyright (c) 2021-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""File source stage."""

import logging
import pathlib
import typing

import mrc

# pylint: disable=morpheus-incorrect-lib-from-import
from morpheus._lib.messages import MessageMeta as CppMessageMeta
from morpheus.cli import register_stage
from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


class CudfSourceStage(PreallocatorMixin, SingleOutputSource):

    def __init__(self, c: Config, num_messages: int = 65536, num_rows: int = 65536):

        super().__init__(c)

        self._num_messages = num_messages
        self._num_rows = num_rows

    @property
    def name(self) -> str:
        return "from-cudf"

    def supports_cpp_node(self) -> bool:
        return True

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:

        import morpheus._lib.stages as _stages
        node = _stages.CudfSourceStage(builder, self.unique_name, self._num_messages, self._num_rows)

        return node
