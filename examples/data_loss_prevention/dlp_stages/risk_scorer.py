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

import functools

import mrc
import numpy as np
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_aliases import SeriesType
from morpheus.utils.type_utils import get_df_class
from morpheus.utils.type_utils import get_df_pkg


@register_stage("risk-scorer")
class RiskScorer(ControlMessageStage):
    """
    Analyzes findings to calculate risk scores and metrics

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    findings_column : str
        Name of the column containing findings to score.
    type_weights : dict[str, int] | None
        Dictionary mapping data types to their risk weights.
        If None, uses `RiskScorer.DEFAULT_TYPE_WEIGHTS`.
    default_weight : int
        Default weight to use for data types not in `type_weights`.
    """

    DEFAULT_TYPE_WEIGHTS = {
        "password": 85,
        "credit_card_number": 90,
        "ssn": 95,
        "street_address": 60,
        "email": 40,
        "phone_number": 45,
        "ipv4": 30,
        "ipv6": 30,
        "date": 20,
        "date_time": 20,
        "time": 20,
        "api_key": 80,
        "customer_id": 65,
        "health_plan_beneficiary_number": 75,
        "medical_record_number": 75
    }

    _NEW_COLUMNS = {
        "risk_score": 0,
        "risk_level": '',
        "highest_confidence": 0.0,
        "num_minimal": 0,
        "num_low": 0,
        "num_medium": 0,
        "num_high": 0,
        "num_critical": 0
    }

    def __init__(self,
                 config: Config,
                 *,
                 findings_column: str,
                 type_weights: dict[str, int] | None = None,
                 default_weight: int = 50):
        """Initialize with configuration for risk scoring"""
        super().__init__(config)

        if type_weights is not None:
            self.type_weights = type_weights
        else:
            self.type_weights = self.DEFAULT_TYPE_WEIGHTS.copy()

        # Default weight if type not in dictionary
        self.default_weight = default_weight

        self._findings_column = findings_column
        self._df_class = get_df_class(config.execution_mode)
        self._df_pkg = get_df_pkg(config.execution_mode)
        self._group_cols = [self._findings_column, "data_types_found"] + list(self._NEW_COLUMNS.keys())

    @property
    def name(self) -> str:
        return "risk-scorer"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    @staticmethod
    def _risk_score_to_level(risk_score: int) -> str:
        """Convert risk score to risk level string"""
        if risk_score >= 80:
            return "critical"

        if risk_score >= 60:
            return "high"

        if risk_score >= 40:
            return "medium"

        if risk_score >= 20:
            return "low"

        return "minimal"

    def _score_fn(self,
                  findings: list[dict] | list[str],
                  *,
                  findings_column: str,
                  type_weights: dict[str, int],
                  default_weight: int,
                  df_class: type) -> DataFrameType:

        # Calculate total weighted score
        total_score = 0
        score_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0, "minimal": 0}

        data_types_found = set()
        highest_confidence = 0

        for finding in findings:
            # When `finding` is a dict it came from the GliNER processor, if not then it was bypassed
            if isinstance(finding, dict):
                data_type: str = finding["label"]

                # Adjust by confidence
                confidence = finding["score"]
            else:
                data_type = finding
                confidence = 1.0

            data_types_found.add(data_type)

            # Get weight for this data type
            weight = type_weights.get(data_type, default_weight)

            if confidence > highest_confidence:
                highest_confidence = confidence

            weighted_score = weight * confidence
            total_score += weighted_score

            # Count by severity
            score_counts[RiskScorer._risk_score_to_level(weight)] += 1

        # Normalize to 0-100 scale with diminishing returns for many findings
        max_score = 100

        # Calculate normalized risk score
        risk_score = round(min(max_score, total_score / len(findings)))

        # Determine risk level from score
        risk_level = RiskScorer._risk_score_to_level(risk_score).title()

        df_data = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "data_types_found": [sorted(data_types_found)],
            "highest_confidence": highest_confidence,
            findings_column: [findings]
        }

        df_data.update({f"num_{level}": count for (level, count) in score_counts.items()})

        return df_class(df_data)

    def _mk_flat(self, findings: SeriesType, *, findings_column: str, df_class: type) -> DataFrameType | None:
        if findings is None:
            return None

        findings = findings.list.leaves

        if len(findings) == 0:
            return None

        return df_class({findings_column: [findings]})

    def score(self, msg: ControlMessage) -> ControlMessage:
        """
        Calculate risk scores based on findings
        """

        with msg.payload().mutable_dataframe() as df:
            findings_ser = df[self._findings_column]
            is_str_col = findings_ser.dtype != 'list'
            if is_str_col:
                # When using --regex_only, the findings column is is a string of comma-separated values, when this
                # is the case we need to split it into a list of unique strings.
                df[self._findings_column] = findings_ser.str.replace(', ', ',', regex=False).str.split(',', regex=False)

            # We split the incoming rows by paragraphs, so we need to group by the original source index and
            # aggregate (flatten) the findings into a single list per source index.
            groups = df.groupby(["original_source_index"], as_index=False)
            flat_fn = functools.partial(self._mk_flat, findings_column=self._findings_column, df_class=self._df_class)
            flat_df = groups[self._findings_column].apply(flat_fn)

        if is_str_col:
            # When using --regex_only we will end up (potentially) with duplicate labels
            # When not using --regex_only, the findings column is a list of dicts, so we don't need to do this.
            flat_df[self._findings_column] = flat_df[self._findings_column].list.unique().list.sort_values()

        # Clean up the resulting DataFrame
        flat_df.index.name = "original_source_index"

        # I'm not sure what this column is, but the value is always 0
        flat_df.drop(columns='index', inplace=True)
        flat_df.reset_index(drop=False, inplace=True)
        flat_df.index.name = "index"

        flat_df = df.assign(**self._NEW_COLUMNS)
        flat_df["data_types_found"] = self._df_pkg.Series(index=df.index, dtype=self._df_pkg.core.dtypes.ListDtype)

        score_fn = functools.partial(self._score_fn,
                                     findings_column=self._findings_column,
                                     type_weights=self.type_weights,
                                     default_weight=self.default_weight,
                                     df_class=self._df_class)
        groups = flat_df.groupby([flat_df.index.name], as_index=False)
        result_df = groups[self._group_cols].apply(score_fn)

        msg.payload(MessageMeta(result_df))

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.score))
        builder.make_edge(input_node, node)

        return node
