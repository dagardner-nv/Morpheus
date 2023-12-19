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

import cupy as cp
import numpy as np
import pytest

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import MessageMeta


def build_post_proc_message(messages_mod, dataset_cudf: DatasetManager, log_test_data_dir: str):
    input_file = os.path.join(TEST_DIRS.validation_data_dir, 'log-parsing-validation-data-input.csv')
    input_df = dataset_cudf[input_file]
    meta = MessageMeta(input_df)

    # we have tensor data for the first five rows
    count = 5
    tensors = {}
    for tensor_name in ['confidences', 'input_ids', 'labels', 'seq_ids']:
        tensor_file = os.path.join(log_test_data_dir, f'{tensor_name}.csv')
        host_data = np.loadtxt(tensor_file, delimiter=',')
        tensors[tensor_name] = cp.asarray(host_data)

    memory = messages_mod.PostprocMemoryLogParsing(count=5, **tensors)
    return messages_mod.MultiPostprocLogParsingMessage(meta=meta,
                                                       mess_offset=0,
                                                       mess_count=count,
                                                       memory=memory,
                                                       offset=0,
                                                       count=count)


@pytest.mark.use_python
@pytest.mark.import_mod([
    os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'messages.py'),
    os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'postprocessing.py')
])
def test_log_parsing_post_processing_stage(config: Config,
                                           dataset_cudf: DatasetManager,
                                           import_mod: typing.List[types.ModuleType]):
    messages_mod, postprocessing_mod = import_mod

    model_vocab_file = os.path.join(TEST_DIRS.data_dir, 'bert-base-cased-vocab.txt')
    log_test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, 'examples/log_parsing')
    model_config_file = os.path.join(log_test_data_dir, 'log-parsing-config.json')

    stage = postprocessing_mod.LogParsingPostProcessingStage(config,
                                                             vocab_path=model_vocab_file,
                                                             model_config_path=model_config_file)

    post_proc_message = build_post_proc_message(messages_mod, dataset_cudf, log_test_data_dir)
    expected_df = dataset_cudf.pandas[os.path.join(log_test_data_dir, 'expected_out.csv')]

    out_meta = stage._postprocess(post_proc_message)

    assert isinstance(out_meta, MessageMeta)
    DatasetManager.assert_compare_df(out_meta._df, expected_df)
