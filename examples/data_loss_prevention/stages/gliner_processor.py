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

import logging
import os
import typing
from functools import partial

import cupy as cp
import mrc
import torch
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages.memory.inference_memory import InferenceMemory
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.utils.type_aliases import DataFrameType

from .gliner_triton import GliNERTritonInference

logger = logging.getLogger(f"morpheus.{__name__}")

EntitiesType = list[dict[str, typing.Any]]
SpanType = tuple[int, int]


@register_stage("gliner-processor")
class GliNERProcessor(ControlMessageStage):
    """
    Process text with a Small Language Model to identify semantically sensitive content
    Uses a model to predict entities in text

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    model_source_dir : str
        Path to the directory containing the GLiNER model files. Used for pre and post-processing.
    server_url : str
        URL of the Triton inference server.
    triton_model_name : str
        Name of the Triton model to use for inference.
    source_column_name : str
        Name of the column containing the source text to process.
    regex_col_prefix : str
        Prefix used to identify regex match columns in the DataFrame.
    confidence_threshold: float
        Minimum confidence score to report a finding
    context_window : int
        Number of characters before and after a regex match to include in the context for SLM analysis.
    fallback : bool
        If True, fallback to GLiNER prediction if no regex findings are available.
        If False, only process rows with regex findings.
    """

    def __init__(self,
                 config: Config,
                 *,
                 model_source_dir: str,
                 server_url: str = "localhost:8001",
                 source_column_name: str = "source_text",
                 confidence_threshold: float = 0.3,
                 fallback: bool = True):

        super().__init__(config)
        self._model = None
        self._model_source_dir = model_source_dir
        self._map_location = map_location
        self._labels_embeddings = None
        self._labels: list[str] | None = None
        self._labels_file = os.path.join(model_source_dir, "label_embedding.pt")

        self._model_max_batch_size = config.model_max_batch_size
        self.source_column_name = source_column_name
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "gliner-processor"

    def accepted_types(self) -> tuple:
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        return False

    @property
    def model(self) -> "GLiNER":
        """
        Return the GLiNER model instance.
        """
        if self._model is None:
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(self._model_source_dir, local_files_only=True, map_location="cuda")
        return self._model

    def _load_label_data(self):
        label_data = torch.load(self._labels_file)
        self._labels_embeddings = cp.asarray(label_data['embeddings'].to("cuda"))
        self._labels = label_data['labels']

    @property
    def labels_embeddings(self) -> cp.ndarray:
        """
        Return the labels embeddings tensor.
        If not loaded, it will load from the specified file.
        """
        if self._labels_embeddings is None:
            self._load_label_data()

        return self._labels_embeddings

    # def _process_results(self, df: DataFrameType, model_entities: list[list[EntitiesType]]) -> list[list[EntitiesType]]:

    #     # flattend the model_entities list
    #     flat_entities = []
    #     for entities in model_entities:
    #         assert entities is not None
    #         flat_entities.extend(entities)

    #     return flat_entities

    # def _infer_callback(self,
    #                     *,
    #                     batch_num: int,
    #                     model_entities: list[list[EntitiesType]],
    #                     future: mrc.Future,
    #                     entities: list[EntitiesType]):
    #     model_entities[batch_num] = entities
    #     future.set_result(batch_num)

    def pre_process(self, context_series):
        """
        Pre-process the data for the ONNX model.
        """
        # === 1. PRE-PROCESSING ===
        model_input, raw_batch = self.model.prepare_model_inputs(context_series.to_pandas(), self.labels, prepare_entities=False)

        # Convert torch tensors to numpy for Triton
        tensors = {
            "input_ids": cp.asarray(model_input["input_ids"]),
            "attention_mask": cp.asarray(model_input["attention_mask"]),
            "words_mask": cp.asarray(model_input["words_mask"]),
            "text_lengths": cp.asarray(model_input["text_lengths"]),
            "span_idx": cp.asarray(model_input["span_idx"]),
            "span_mask": cp.asarray(model_input["span_mask"])
        }

        return tensors, raw_batch

    def process(self, msg: ControlMessage) -> ControlMessage:
        """
        Analyze text using an entity prediction model for sensitive data detection
        """

        with msg.payload().mutable_dataframe() as df:
            context_series = df['context']
            (tensors, raw_batch) = self.pre_process(context_series)
            memory = InferenceMemory(len(df), tensors=tensors)
            memory.set_tensor("labels_embeddings", self.labels_embeddings)

        msg.set_metadata("gliner_raw_batch", raw_batch)
        msg.te

        # futures = []
        # model_entities = []
        # for i in range(0, len(df), self._model_max_batch_size):
        #     future = mrc.Future()
        #     futures.append(future)
        #     model_entities.append(None)
        #     batch_data = context_series[i:i + self._model_max_batch_size]

        #     self.gliner_triton.process(
        #         batch_data.to_arrow().to_pylist(),
        #         partial(self._infer_callback,
        #                 batch_num=len(model_entities) - 1,
        #                 model_entities=model_entities,
        #                 future=future))

        # for future in futures:
        #     future.result()

        # dlp_findings = self._process_results(df, model_entities)

        # df['dlp_findings'] = dlp_findings

        return msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.process))
        builder.make_edge(input_node, node)

        return node
