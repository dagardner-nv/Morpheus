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

import json
import logging
import pathlib

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.utils.type_utils import get_df_class
from morpheus.utils.type_utils import get_df_pkg

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("regex-processor")
class RegexProcessor(GpuAndCpuMixin, ControlMessageStage):
    """
    Process text with regex patterns to identify structured sensitive data

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    patterns: dict[str, list[str]] | None
        Dictionary mapping data types to lists of regex patterns
    patterns_file : str | pathlib.Path | None
        Path to a JSON file containing regex patterns for different data types.
        Ignored if `patterns` is provided.
    source_column_name : str
        Name of the column containing the source text to process.
    """

    def __init__(self,
                 config: Config,
                 *,
                 patterns: dict[str, list[str]] | None = None,
                 patterns_file: str | pathlib.Path | None = None,
                 source_column_name: str = "source_text",
                 context_window: int = 100):
        """
        Initialize with regex patterns to detect sensitive data

        Args:
            patterns: Dictionary mapping data types to lists of regex patterns
            case_sensitive: Whether regex matching should be case sensitive
        """
        super().__init__(config)
        self.source_column_name = source_column_name
        self.context_window = context_window
        self.combined_patterns = {}
        self._df_pkg = get_df_pkg(config.execution_mode)
        self._df_class = get_df_class(config.execution_mode)

        if patterns is None:
            if patterns_file is None:
                raise ValueError("Either 'patterns' or 'patterns_file' must be provided")
            patterns = self.load_regex_patterns(patterns_file)
            logger.info("Loaded %d regex pattern groups", len(patterns))

        # For each entity type, combine multiple patterns into a single regex
        for pattern_name, pattern_list in patterns.items():

            # Combine all patterns for this entity type with OR operator
            if len(pattern_list) > 1:
                combined_pattern = '|'.join(f'(?:{p})' for p in pattern_list)
            else:
                combined_pattern = pattern_list[0]

            self.combined_patterns[pattern_name] = combined_pattern

    @staticmethod
    def load_regex_patterns(file_path: str | pathlib.Path) -> dict[str, list[str]]:
        """Load regex patterns from a JSON file."""
        with open(file_path, 'r', encoding="utf-8") as f:
            return json.load(f)

    @property
    def name(self) -> str:
        return "regex-processor"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    @property
    def patterns(self) -> dict[str, str]:
        """
        Returns the compiled regex patterns used for detection.
        """
        return self.combined_patterns.copy()

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Scan text for sensitive data using regex patterns

        Returns:
            List of findings with metadata
        """

        with msg.payload().mutable_dataframe() as df:
            # Extract the text column to process
            if df.index.name is None:
                df.index.name = "original_row"  # Ensure index has a name for consistency

            text_series = df[self.source_column_name]

            matched_dfs = []
            for pattern_name, pattern in self.combined_patterns.items():
                matched_series = text_series.str.findall(pattern)
                matched_series = matched_series.explode(ignore_index=False).dropna()
                if len(matched_series) > 0:
                    matched_dfs.append(self._df_class({
                        'matches': matched_series,
                        'pattern_name': pattern_name,
                    }))

            #         temp_df = temp_df.merge(df, on=[df.index.name])

            #         span_start = temp_df[self.source_column_name].str.find_multiple(matched_series).explode(
            #             ignore_index=False)
            #         span_start.replace(-1, None, inplace=True)  # Replace -1 with None for no match
            #         span_start.dropna(inplace=True)

            #         print(f"temp_df ({len(temp_df)}):\n{temp_df}\n\nspan_start({len(span_start)}):\n{span_start}")

            #         temp_df = temp_df.merge(self._df_class({"span_start": span_start}), on=[df.index.name])
            #         temp_df['span_end'] = temp_df['span_start'] + temp_df['matches'].str.len()

            #         matched_dfs.append(temp_df)

            # merged_df = self._df_pkg.concat(matched_dfs)
            # merged_df.reset_index(drop=False, inplace=True)
            # context_start = (merged_df['span_start'] - self.context_window).clip(lower=0)
            # context_end = merged_df['span_end'] + self.context_window
            # merged_df['context_start'] = context_start
            # merged_df['context_end'] = context_end
            # merged_df['source_text_length'] = merged_df[self.source_column_name].str.len()
            # merged_df['context_end'] = merged_df[['context_end', 'source_text_length']].min(axis=1)
            # merged_df.drop(columns=['source_text_length'], inplace=True)
            # merged_df['context'] = merged_df[self.source_column_name].str.slice_from(
            #     merged_df['context_start'], merged_df['context_end'])

            matches_df = self._df_pkg.concat(matched_dfs)
            merged_df = df.merge(matches_df, on=[df.index.name])

            new_meta = MessageMeta(merged_df)
            msg.payload(new_meta)

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
