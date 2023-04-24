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

import glob
import os
import types
import typing

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.messages.message_meta import AppShieldMessageMeta
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.stages.input.appshield_source_stage import AppShieldSourceStage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage
from utils import TEST_DIRS
from utils.dataset_manager import DatasetManager


@pytest.mark.use_python
class TestPreprocessingRWStage:

    def test_constructor(self, config: Config, rwd_conf: dict):
        from stages.preprocessing import PreprocessingRWStage

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=6)
        assert stage._feature_columns == rwd_conf['model_features']
        assert stage._features_len == len(rwd_conf['model_features'])
        assert stage._snapshot_dict == {}
        assert len(stage._padding_data) == len(rwd_conf['model_features']) * 6
        for i in stage._padding_data:
            assert i == 0

    def test_sliding_window_offsets(self, config: Config, rwd_conf: dict):
        from stages.preprocessing import PreprocessingRWStage

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=6)

        window = 3
        ids = [17, 18, 19, 20, 21, 22, 23, 31, 32, 33]
        results = stage._sliding_window_offsets(ids, len(ids), window=window)
        assert results == [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (7, 10)]

        # Non-consecutive ids don't create sliding windows
        stage._sliding_window_offsets(list(reversed(ids)), len(ids), window=window) == []

    def test_sliding_window_offsets_errors(self, config: Config, rwd_conf: dict):
        from stages.preprocessing import PreprocessingRWStage

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=6)

        # ids_len doesn't match the length of the ids list
        with pytest.raises(AssertionError):
            stage._sliding_window_offsets(ids=[5, 6, 7], ids_len=12, window=2)

        # Window is larger than ids
        with pytest.raises(AssertionError):
            stage._sliding_window_offsets(ids=[5, 6, 7], ids_len=3, window=4)

    def test_rollover_pending_snapshots(self, config: Config, rwd_conf: dict, dataset_pandas: DatasetManager):
        from stages.preprocessing import PreprocessingRWStage

        snapshot_ids = [5, 8, 10, 13]
        source_pid_process = "123_test.exe"
        df = dataset_pandas['examples/ransomware_detection/dask_results.csv']
        assert len(df) == len(snapshot_ids)

        # The snapshot_id's in the test data set are all '1', set them to different values
        df['snapshot_id'] = snapshot_ids
        df.index = df.snapshot_id

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=4)
        stage._rollover_pending_snapshots(snapshot_ids, source_pid_process, df)

        assert list(stage._snapshot_dict.keys()) == [source_pid_process]

        # Due to the sliding window we should have all but the first snapshot_id in the results
        expected_snapshot_ids = snapshot_ids[1:]
        snapshots = stage._snapshot_dict[source_pid_process]

        assert len(snapshots) == len(expected_snapshot_ids)
        for (i, snapshot) in enumerate(snapshots):
            expected_snapshot_id = expected_snapshot_ids[i]
            assert snapshot.snapshot_id == expected_snapshot_id
            expected_data = df.loc[expected_snapshot_id].fillna('').values
            assert (pd.Series(snapshot.data).fillna('').values == expected_data).all(), f"Data for {expected_snapshot_id} does not match"

    def test_rollover_pending_snapshots_empty_results(self,
                                                      config: Config,
                                                      rwd_conf: dict,
                                                      dataset_pandas: DatasetManager):
        from stages.preprocessing import PreprocessingRWStage

        snapshot_ids = []
        source_pid_process = "123_test.exe"
        df = dataset_pandas['examples/ransomware_detection/dask_results.csv']

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=4)
        stage._rollover_pending_snapshots(snapshot_ids, source_pid_process, df)
        assert len(stage._snapshot_dict) == 0
