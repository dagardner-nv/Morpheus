# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
"""Sourse stage for Duo Authentication logs."""

import json
import logging
import typing

import pandas as pd

from morpheus.cli import register_stage
from morpheus.config import PipelineModes
from morpheus.stages.input.autoencoder_source_stage import AutoencoderSourceStage

logger = logging.getLogger(__name__)


@register_stage("from-duo", modes=[PipelineModes.AE])
class DuoSourceStage(AutoencoderSourceStage):
    """
    Source stage is used to load Duo Authentication messages.

    Adds the following derived features:
        - `locincrement`: Increments every time a log contains a distinct city within a day.
        - `logcount`: Tracks the number of logs generated by a user within a day.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    input_glob : str
        Input glob pattern to match files to read. For example, `./input_dir/*.json` would read all files with the
        'json' extension in the directory input_dir.
    watch_directory : bool, default = False
        The watch directory option instructs this stage to not close down once all files have been read. Instead it will
        read all files that match the 'input_glob' pattern, and then continue to watch the directory for additional
        files. Any new files that are added that match the glob will then be processed.
    max_files: int, default = -1
        Max number of files to read. Useful for debugging to limit startup time. Default value of -1 is unlimited.
    file_type : `morpheus.common.FileTypes`, default = 'FileTypes.Auto'.
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'json', 'csv'
    repeat: int, default = 1
        How many times to repeat the dataset. Useful for extending small datasets in debugging.
    sort_glob : bool, default = False
        If true the list of files matching `input_glob` will be processed in sorted order.
    recursive: bool, default = True
        If true, events will be emitted for the files in subdirectories that match `input_glob`.
    queue_max_size: int, default = 128
        Maximum queue size to hold the file paths to be processed that match `input_glob`.
    batch_timeout: float, default = 5.0
        Timeout to retrieve batch messages from the queue.
    """

    @property
    def name(self) -> str:
        """Unique name for the stage."""
        return "from-duo"

    def supports_cpp_node(self):
        """Indicate that this stages does not support a C++ node."""
        return False

    @staticmethod
    def change_columns(df):
        """
        Removes characters (_,.,{,},:) from the names of the dataframe columns.

        Parameters
        ----------
        df : `pd.DataFrame`
            Dataframe that requires column renaming.

        Returns
        -------
        df : `pd.DataFrame`
            Dataframe with renamed columns.
        """
        df.columns = df.columns.str.replace('[_,.,{,},:]', '')
        df.columns = df.columns.str.strip()
        return df

    @staticmethod
    def derive_features(df: pd.DataFrame, feature_columns: typing.List[str]):
        """
        Derives feature columns from the DUO (logs) source columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for deriving columns.
        feature_columns : typing.List[str]
            Names of columns that are need to be derived.

        Returns
        -------
        df : typing.List[pd.DataFrame]
            Dataframe with actual and derived columns.
        """
        _DEFAULT_DATE = '1970-01-01T00:00:00.000000+00:00'
        timestamp_column = "isotimestamp"
        city_column = "accessdevicelocationcity"
        state_column = "accessdevicelocationstate"
        country_column = "accessdevicelocationcountry"

        df['time'] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df['day'] = df['time'].dt.date
        df.fillna({'time': pd.to_datetime(_DEFAULT_DATE), 'day': pd.to_datetime(_DEFAULT_DATE).date()}, inplace=True)
        df.sort_values(by=['time'], inplace=True)

        overall_location_columns = [col for col in [city_column, state_column, country_column] if col is not None]
        overall_location_df = df[overall_location_columns].fillna('nan')
        df['overall_location'] = overall_location_df.apply(lambda x: ', '.join(x), axis=1)
        df['loc_cat'] = df.groupby('day')['overall_location'].transform(lambda x: pd.factorize(x)[0] + 1)
        df.fillna({'loc_cat': 1}, inplace=True)
        df['locincrement'] = df.groupby('day')['loc_cat'].expanding(1).max().droplevel(0)
        df.drop(['overall_location', 'loc_cat'], inplace=True, axis=1)

        df["logcount"] = df.groupby('day').cumcount()

        if (feature_columns is not None):
            df.drop(columns=df.columns.difference(feature_columns), inplace=True)

        return df

    @staticmethod
    def files_to_dfs_per_user(x: typing.List[str],
                              userid_column_name: str,
                              feature_columns: typing.List[str],
                              userid_filter: str = None,
                              repeat_count: int = 1) -> typing.Dict[str, pd.DataFrame]:
        """
        After loading the input batch of DUO logs into a dataframe, this method builds a dataframe
        for each set of userid rows in accordance with the specified filter condition.

        Parameters
        ----------
        x : typing.List[str]
            List of messages.
        userid_column_name : str
            Name of the column used for categorization.
        feature_columns : typing.List[str]
            Feature column names.
        userid_filter : str
            Only rows with the supplied userid are filtered.
        repeat_count : str
            Number of times the given rows should be repeated.

        Returns
        -------
        df_per_user  : typing.Dict[str, pd.DataFrame]
            Dataframe per userid.
        """
        dfs = []
        for file in x:
            with open(file, encoding='UTF-8') as json_in:
                log = json.load(json_in)
            df = pd.json_normalize(log)
            df = DuoSourceStage.change_columns(df)
            dfs = dfs + AutoencoderSourceStage.repeat_df(df, repeat_count)

        df_per_user = AutoencoderSourceStage.batch_user_split(dfs, userid_column_name, userid_filter)

        return df_per_user
