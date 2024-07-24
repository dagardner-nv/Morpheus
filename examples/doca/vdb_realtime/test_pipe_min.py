# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

import os

import click

from morpheus.cli.utils import get_log_levels
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.messages import RawPacketMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.doca.doca_convert_stage import DocaConvertStage
from morpheus.stages.doca.doca_source_stage import DocaSourceStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.utils.logger import configure_logging
from morpheus.cli.utils import parse_log_level


@click.command()
@click.option(
    "--nic_addr",
    help="NIC PCI Address",
    required=True,
)
@click.option(
    "--gpu_addr",
    help="GPU PCI Address",
    required=True,
)
@click.option("--convert", type=bool, default=False, is_flag=True)
@click.option("--net_type", type=str, default='udp')
@click.option("--log_level",
              default="INFO",
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              show_default=True,
              help="Specify the logging level to use.")
@click.option("--out_file", type=str, default=None)
@click.option("--buffer_size", type=int, default=1024)
@click.option("--buffer_secs", type=int, default=3)
def run_pipeline(nic_addr,
                 gpu_addr,
                 log_level: int,
                 out_file: str,
                 net_type: str,
                 convert: bool,
                 buffer_size: int,
                 buffer_secs: int):
    morpheus_root = os.environ.get('MORPHEUS_ROOT')

    # Enable the default logger
    configure_logging(log_level=log_level)

    CppConfig.set_should_use_cpp(True)

    config = Config()
    config.mode = PipelineModes.NLP
    config.pipeline_batch_size = 1024
    config.feature_length = 512
    config.edge_buffer_size = 512
    config.num_threads = 20

    pipeline = LinearPipeline(config)

    # add doca source stage
    pipeline.set_source(DocaSourceStage(config, nic_addr, gpu_addr, net_type))

    if convert:
        pipeline.add_stage(DocaConvertStage(config, buffer_channel_size=buffer_size, max_time_delta_sec=buffer_secs))
        print("Using DocaConvertStage")
        pipeline.add_stage(MonitorStage(config, description="Doca Convert Rate", unit='pkts', delayed_start=True))
    else:
        print("DocaSourceStage only")

        def count_raw_packets(message: RawPacketMessage):
            return message.num

        pipeline.add_stage(
            MonitorStage(config,
                         description="DOCA GPUNetIO Source rate",
                         unit='pkts',
                         determine_count_fn=count_raw_packets,
                         delayed_start=True))

    if out_file is not None:
        out_file_path = os.path.join(morpheus_root, ".tmp", out_file)
        print(f"Writing to: {out_file_path}")
        pipeline.add_stage(WriteToFileStage(config, filename=out_file_path, overwrite=True, flush=True))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
