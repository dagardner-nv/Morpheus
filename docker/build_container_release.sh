#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script file to ensure we are in the correct repo (in case were in a submodule)
pushd ${SCRIPT_DIR} &> /dev/null

export DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"nvcr.io/nvidia/morpheus/morpheus"}
export DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"$(git describe --tags --abbrev=0)-runtime"}
export DOCKER_TARGET=${DOCKER_TARGET:-"runtime"}

# For the release container we copy several files into the container, in order to avoid including files not in the repo
# we use git ls-files to get the list of files to copy
pushd ${SCRIPT_DIR}/../ &> /dev/null

_COPY_FILES=()
_COPY_FILES+=$(git ls-files conda/environments/*.yaml)
_COPY_FILES+=" "
_COPY_FILES+=$(git ls-files docker)
_COPY_FILES+=" "
_COPY_FILES+=$(git ls-files docs)
_COPY_FILES+=" "
_COPY_FILES+=$(git ls-files examples)
_COPY_FILES+=" "
_COPY_FILES+=$(git ls-files models)
_COPY_FILES+=" "
_COPY_FILES+=$(git ls-files scripts)
_COPY_FILES+=" "
_COPY_FILES+=$(git ls-files *.md)
_COPY_FILES+=" "
_COPY_FILES+=("LICENSE")
popd &> /dev/null

export COPY_FILES=$(echo ${_COPY_FILES[@]} | tr '\n' ' ')

popd &> /dev/null

# Fetch data
"${SCRIPT_DIR}/../scripts/fetch_data.py" fetch docs examples models

# Call the general build script
${SCRIPT_DIR}/build_container.sh
