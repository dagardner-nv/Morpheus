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


import cudf
import pytest
from unittest.mock import patch
from morpheus.stages.output.write_to_databricks_deltalake_stage import DataBricksDeltaLakeSinkStage
from _utils.dataset_manager import DatasetManager
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

@pytest.mark.use_cudf
def test_databricks_deltalake_source_stage_pipe(config: Config, dataset: DatasetManager):
    """
    Test the DataBricksDeltaLakeSinkStage against a mock spark session which
    will create spark dataframe that will be written to remote
    location from databricks cluster.
    """

    df_input_a = dataset['filter_probs.csv']
    with patch('utils.write_to_databricks_deltalake_stage.DatabricksSession',) as mock_db_session:
        databricks_deltalake_sink_stage = DataBricksDeltaLakeSinkStage(config,
                     delta_path="", delta_table_write_mode="append",
                     databricks_host="", databricks_token="",
                     databricks_cluster_id="")

        # df_input_a = cudf.DataFrame({"name": ["five", "four", "three", "two", "one"], "value": [5, 4, 3, 2, 1]})
        pipeline = LinearPipeline(config)
        pipeline.set_source(InMemorySourceStage(config, [df_input_a]))
        pipeline.add_stage(DeserializeStage(config))
        pipeline.add_stage(SerializeStage(config))
        sink = pipeline.add_stage(databricks_deltalake_sink_stage)
        pipeline.run()
        databricks_deltalake_sink_stage.spark.createDataFrame.assert_called_once()