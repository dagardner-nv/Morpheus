#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

source ${WORKSPACE}/ci/scripts/github/common.sh

update_conda_env

rapids-logger "Check versions"
python3 --version
gcc --version
g++ --version
cmake --version
ninja --version

rapids-logger "Configuring cmake for Morpheus"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} \
    -DCCACHE_PROGRAM_PATH=$(which sccache) .

rapids-logger "Building Morpheus"
cmake --build build --parallel ${PARALLEL_LEVEL}

rapids-logger "sccache usage for morpheus build:"
sccache --show-stats

rapids-logger "Installing Morpheus"
cmake -DCOMPONENT=Wheel -P ${MORPHEUS_ROOT}/build/cmake_install.cmake

rapids-logger "Archiving results"
tar cfj "${WORKSPACE_TMP}/wheel.tar.bz" build/wheel

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${WORKSPACE_TMP}/wheel.tar.bz" "${ARTIFACT_URL}/wheel.tar.bz"

rapids-logger "Success"
exit 0
