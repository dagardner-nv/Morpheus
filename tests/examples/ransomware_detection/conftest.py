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
import sys

import pytest
import yaml

from utils import TEST_DIRS

SKIP_REASON = ("Tests for the ransomware_detection example require a number of packages not installed in the Morpheus "
               "development environment. See `/home/dagardner/work/morpheus/examples/ransomware_detection/README.md` "
               "for details on installing these additional dependencies")


@pytest.fixture(autouse=True, scope='session')
def dask_distributed():
    """
    All of the tests in this subdir requires dask.distributed
    """
    yield pytest.importorskip("dask.distributed", reason=SKIP_REASON)


@pytest.fixture
def config(config):
    """
    The ransomware detection pipeline utilizes the FIL pipeline mode.
    """
    from morpheus.config import PipelineModes
    config.mode = PipelineModes.FIL
    yield config


@pytest.fixture
def example_dir():
    yield os.path.join(TEST_DIRS.examples_dir, 'ransomware_detection')


@pytest.fixture
def conf_file(example_dir):
    yield os.path.join(example_dir, 'config/ransomware_detection.yaml')


@pytest.fixture
def rwd_conf(conf_file):
    with open(conf_file, encoding='UTF-8') as fh:
        conf = yaml.safe_load(fh)

    yield conf


@pytest.fixture
def interested_plugins():
    yield ['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']


@pytest.fixture
@pytest.mark.use_pandas
def df_with_pids(dataset_pandas):
    """
    Inserts a new column `pid_process` into the `filter_probs_df` dataframe. The values consist of 6 unique randomly
    generated integers in the range of plausable process ids on a typical Linux system.

    Note the pids here are intentionally un-sorted, much of the logic in this pipeline requires this DF to be sorted.
    """
    df = dataset_pandas["filter_probs.csv"]
    df['pid_process'] = [
        2788672,
        75956,
        75956,
        2788672,
        2788672,
        2698363,
        2721362,
        118469,
        1348612,
        2698363,
        118469,
        2698363,
        1348612,
        118469,
        75956,
        2721362,
        75956,
        118469,
        118469,
        118469
    ]

    yield df


# Some of the code inside ransomware_detection performs imports in the form of:
#    from common....
# For this reason we need to ensure that the examples/ransomware_detection dir is in the sys.path first
@pytest.fixture(autouse=True)
def ransomware_detection_in_sys_path(request: pytest.FixtureRequest, restore_sys_path, reset_plugins, example_dir):
    sys.path.append(example_dir)
