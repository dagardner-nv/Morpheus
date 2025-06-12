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

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.utils.type_utils import get_df_pkg

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("gliner-preprocess")
class GliNERPreprocess(GpuAndCpuMixin, ControlMessageStage):

    def __init__(self, config: Config, *, source_column_name: str = "source_text", context_window: int = 100):

        super().__init__(config)
        self.source_column_name = source_column_name
        self.context_window = context_window
        self._df_pkg = get_df_pkg(config.execution_mode)

    @property
    def name(self) -> str:
        return "gliner-preprocess"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    def extract_spans(self, row: pd.Series) -> list[typing.Any]:
        src_txt = row.source_text
        match = row.matches
        span_start = src_txt.find(match)
        span_end = span_start + len(match)
        context_start = max(0, span_start - self.context_window)
        context_end = min(len(src_txt), span_end + self.context_window)
        context = src_txt[context_start:context_end]
        return [span_start, span_end, context_start, context_end, context]

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Analyze text using an entity prediction model for sensitive data detection
        """

        with msg.payload().mutable_dataframe() as df:
            is_pandas = isinstance(df, pd.DataFrame)
            if not is_pandas:
                working_df = df.to_pandas()
            else:
                working_df = df

            results = working_df.apply(self.extract_spans, axis=1, result_type='expand')
            if not is_pandas:
                results = self._df_pkg.from_pandas(results)

            df[['span_start', 'span_end', 'context_start', 'context_end', 'context']] = results

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
