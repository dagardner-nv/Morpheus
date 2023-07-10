# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import dataclasses
import typing
from functools import partial

import networkx as nx
import nvtabular as nvt
import pandas as pd
from merlin.dag import ColumnSelector
from nvtabular.ops import Filter
from nvtabular.ops import LambdaOp
from nvtabular.ops import Rename

import cudf

from morpheus.utils.column_info import BoolColumn
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import StringJoinColumn
from morpheus.utils.nvt import MutateOp
from morpheus.utils.nvt.transforms import json_flatten
from morpheus.utils.type_aliases import DataFrameType


def sync_df_as_pandas(func: typing.Callable) -> typing.Callable:
    """
    This function serves as a decorator that synchronizes cudf.DataFrame to pandas.DataFrame before applying the
    function.

    :param func: The function to apply to the DataFrame
    :return: The wrapped function
    """

    def wrapper(df: DataFrameType, **kwargs) -> DataFrameType:
        convert_to_cudf = False
        if isinstance(df, cudf.DataFrame):
            convert_to_cudf = True
            df = df.to_pandas()

        df = func(df, **kwargs)

        if convert_to_cudf:
            df = cudf.from_pandas(df)

        return df

    return wrapper


@dataclasses.dataclass
class JSONFlattenInfo(ColumnInfo):
    """Subclass of `ColumnInfo`, Dummy ColumnInfo -- Makes it easier to generate a graph of the column dependencies"""
    input_col_names: list
    output_col_names: list


def resolve_json_output_columns(input_schema: DataFrameInputSchema) -> typing.List[typing.Tuple[str, str]]:
    """
    Resolves JSON output columns from an input schema.

    :param input_schema: The input schema to resolve the JSON output columns from
    :return: A list of tuples where each tuple is a pair of column name and its data type
    """

    column_info_objects = input_schema.column_info

    json_output_candidates = []
    for col_info in column_info_objects:
        json_output_candidates.append((col_info.name, col_info.dtype))
        if (hasattr(col_info, 'input_name')):
            json_output_candidates.append((col_info.input_name, col_info.dtype))
        if (hasattr(col_info, 'input_columns')):
            for col_name in col_info.input_columns:
                json_output_candidates.append((col_name, col_info.dtype))

    output_cols = []
    json_cols = input_schema.json_columns
    for col in json_output_candidates:
        cnsplit = col[0].split('.')
        if (len(cnsplit) > 1 and cnsplit[0] in json_cols):
            output_cols.append(col)

    return output_cols


def get_ci_column_selector(col_info: ColumnInfo):
    """
    Return a column selector based on a ColumnInfo object.

    :param col_info: The ColumnInfo object
    :return: A column selector
    """
    if (col_info.__class__ == ColumnInfo):
        return col_info.name

    if col_info.__class__ in [RenameColumn, BoolColumn, DateTimeColumn, StringJoinColumn, IncrementColumn]:
        return col_info.input_name

    if col_info.__class__ == StringCatColumn:
        return col_info.input_columns

    if col_info.__class__ == JSONFlattenInfo:
        return col_info.input_col_names

    if col_info.__class__ == CustomColumn:
        return '*'

    raise ValueError(f"Unknown ColumnInfo type: {col_info.__class__}")


def json_flatten_from_input_schema(json_input_cols: typing.List[str], json_output_cols: typing.List[str]) -> MutateOp:
    """
    Return a JSON flatten operation from an input schema.

    :param json_input_cols: A list of JSON input columns
    :param json_output_cols: A list of JSON output columns
    :return: A MutateOp object that represents the JSON flatten operation
    """
    json_flatten_op = MutateOp(json_flatten, dependencies=json_input_cols, output_columns=json_output_cols)

    return json_flatten_op


@sync_df_as_pandas
def string_cat_col(df: DataFrameType, output_column: str, sep: str) -> DataFrameType:
    """
    Concatenate the string representation of all supplied columns in a DataFrame.

    :param df: The input DataFrame
    :param output_column: The name of the output column
    :param sep: The separator to use when concatenating the strings
    :return: The resulting DataFrame
    """
    cat_col = df.apply(lambda row: sep.join(row.values.astype(str)), axis=1)

    return pd.DataFrame({output_column: cat_col})


def nvt_string_cat_col(
        column_selector: ColumnSelector,  # pylint: disable=unused-argument
        df: DataFrameType,
        output_column: str,
        input_columns: typing.List[str],
        sep: str = ', '):
    """
    Concatenates the string representation of the specified columns in a DataFrame.

    :param column_selector: A ColumnSelector object -> unused.
    :param df: The input DataFrame
    :param output_column: The name of the output column
    :param input_columns: The input columns to concatenate
    :param sep: The separator to use when concatenating the strings
    :return: The resulting DataFrame
    """
    return string_cat_col(df[input_columns], output_column=output_column, sep=sep)


@sync_df_as_pandas
def increment_column(df: DataFrameType, output_column: str, input_column: str, period: str = 'D') -> DataFrameType:
    """
    Crete an increment a column in a DataFrame.

    :param df: The input DataFrame
    :param output_column: The name of the output column
    :param input_column: The name of the input column
    :param period: The period to increment by
    :return: The resulting DataFrame
    """
    period_index = pd.to_datetime(df[input_column]).dt.to_period(period)
    groupby_col = df.groupby([output_column, period_index]).cumcount()

    return pd.DataFrame({output_column: groupby_col})


def nvt_increment_column(column_selector: ColumnSelector,
                         df: DataFrameType,
                         output_column: str,
                         input_column: str,
                         period: str = 'D'):
    return increment_column(column_selector=column_selector,
                            df=df,
                            output_column=output_column,
                            input_column=input_column,
                            period=period)


# Mappings from ColumnInfo types to functions that create the corresponding NVT operator
ColumnInfoProcessingMap = {
    BoolColumn:
        lambda ci,
        deps: [
            LambdaOp(
                lambda series: series.map(ci.value_map).astype(bool), dtype="bool", label=f"[BoolColumn] '{ci.name}'")
        ],
    # ColumnInfo: lambda ci, deps: [
    #    LambdaOp(lambda series: series.astype(ci.dtype), dtype=ci.dtype, label=f"[ColumnInfo] '{ci.name}'")],
    ColumnInfo:
        lambda ci,
        deps: [
            MutateOp(lambda selector,
                     df: df.assign(**{ci.name: df[ci.name].astype(ci.get_pandas_dtype())}) if (ci.name in df.columns)
                     else df.assign(**{ci.name: pd.Series(None, index=df.index, dtype=ci.get_pandas_dtype())}),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[ColumnInfo] '{ci.name}'")
        ],
    # TODO(Devin): Custom columns are, potentially, very inefficient, because we have to run the custom function on the
    #   entire dataset this is because NVT requires the input column be available, but CustomColumn is a generic
    #   transform taking df->series(ci.name)
    CustomColumn:
        lambda ci,
        deps: [
            MutateOp(lambda selector,
                     df: cudf.DataFrame({ci.name: ci.process_column_fn(df)}),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[CustomColumn] '{ci.name}'")
        ],
    DateTimeColumn:
        lambda ci,
        deps: [
            Rename(f=lambda name: ci.name if name == ci.input_name else name),
            LambdaOp(lambda series: series.astype(ci.dtype), dtype=ci.dtype, label=f"[DateTimeColumn] '{ci.name}'")
        ],
    IncrementColumn:
        lambda ci,
        deps: [
            MutateOp(partial(
                nvt_increment_column, output_column=ci.groupby_column, input_column=ci.name, period=ci.period),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.groupby_column)],
                     label=f"[IncrementColumn] '{ci.name}' => '{ci.groupby_column}'")
        ],
    RenameColumn:
        lambda ci,
        deps: [
            MutateOp(lambda selector,
                     df: df.rename(columns={ci.input_name: ci.name}),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[RenameColumn] '{ci.input_name}' => '{ci.name}'")
        ],
    StringCatColumn:
        lambda ci,
        deps: [
            MutateOp(partial(nvt_string_cat_col, output_column=ci.name, input_columns=ci.input_columns, sep=ci.sep),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[StringCatColumn] '{','.join(ci.input_columns)}' => '{ci.name}'")
        ],
    StringJoinColumn:
        lambda ci,
        deps: [
            MutateOp(partial(
                nvt_string_cat_col, output_column=ci.name, input_columns=[ci.name, ci.input_name], sep=ci.sep),
                     dependencies=deps,
                     output_columns=[(ci.name, ci.dtype)],
                     label=f"[StringJoinColumn] '{ci.input_name}' => '{ci.name}'")
        ],
    JSONFlattenInfo:
        lambda ci,
        deps: [json_flatten_from_input_schema(ci.input_col_names, ci.output_col_names)]
}


def build_nx_dependency_graph(column_info_objects: typing.List[ColumnInfo]) -> nx.DiGraph:
    graph = nx.DiGraph()

    def find_dependent_column(name, current_name):
        for col_info in column_info_objects:
            if col_info.name == current_name:
                continue
            if col_info.name == name:
                return col_info
            if col_info.__class__ == JSONFlattenInfo:
                if name in [c for c, _ in col_info.output_col_names]:
                    return col_info
        return None

    for col_info in column_info_objects:
        graph.add_node(col_info.name)

        if col_info.__class__ in [RenameColumn, BoolColumn, DateTimeColumn, StringJoinColumn, IncrementColumn]:
            # If col_info.name != col_info.input_name then we're creating a potential dependency
            if col_info.name != col_info.input_name:
                dep_col_info = find_dependent_column(col_info.input_name, col_info.name)
                if dep_col_info:
                    # This CI is dependent on the dep_col_info CI
                    graph.add_edge(dep_col_info.name, col_info.name)

        elif col_info.__class__ == StringCatColumn:
            for input_col_name in col_info.input_columns:
                dep_col_info = find_dependent_column(input_col_name, col_info.name)
                if dep_col_info:
                    graph.add_edge(dep_col_info.name, col_info.name)

        elif col_info.__class__ == JSONFlattenInfo:
            for output_col_name in [c for c, _ in col_info.output_col_names]:
                dep_col_info = find_dependent_column(output_col_name, col_info.name)
                if dep_col_info:
                    graph.add_edge(dep_col_info.name, col_info.name)

    return graph


def bfs_traversal_with_op_map(graph, ci_map, root_nodes):
    visited = set()
    queue = list(root_nodes)
    node_op_map = {}

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)

            parents = list(graph.predecessors(node))
            if len(parents) == 0:
                # We need to start an operator chain with a column selector, so root nodes need to prepend a parent
                #   column selection operator
                parent_input = get_ci_column_selector(ci_map[node])
            else:
                # Not a root node, so we need to gather the parent operators, and collect them up.
                parent_input = None
                for parent in parents:
                    if parent_input is None:
                        parent_input = node_op_map[parent]
                    else:
                        parent_input = parent_input + node_op_map[parent]

            # Map the column info object to its NVT operator implementation
            ops = ColumnInfoProcessingMap[type(ci_map[node])](ci_map[node], deps=[])

            # Chain ops together into a compound op
            node_op = parent_input
            for operator in ops:
                node_op = node_op >> operator

            # Set the op for this node to the compound operator
            node_op_map[node] = node_op

            # Add our neighbors to the queue
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                queue.append(neighbor)

    return visited, node_op_map


def coalesce_leaf_nodes(node_op_map, graph, preserve_re):
    coalesced_workflow = None
    for node, operator in node_op_map.items():
        neighbors = list(graph.neighbors(node))
        # Only add the operators for leaf nodes, or those explicitly preserved
        if len(neighbors) == 0 or (preserve_re and preserve_re.match(node)):
            if coalesced_workflow is None:
                coalesced_workflow = operator
            else:
                coalesced_workflow = coalesced_workflow + operator

    return coalesced_workflow


def coalesce_ops(graph, ci_map, preserve_re=None):
    root_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]
    _, node_op_map = bfs_traversal_with_op_map(graph, ci_map, root_nodes)
    coalesced_workflow = coalesce_leaf_nodes(node_op_map, graph, preserve_re=preserve_re)

    return coalesced_workflow


def dataframe_input_schema_to_nvt_workflow(input_schema: DataFrameInputSchema, visualize=False) -> nvt.Workflow:
    """
    Converts an `input_schema` to a `nvt.Workflow` object

    First we aggregate all preprocessing steps, which we assume are independent of each other and can be run in
    parallel.

    Next we aggregate all column operations, which we assume are independent of each other and can be run in parallel
    and pass them the updated schema from the preprocessing steps.
    """

    if (input_schema is None or len(input_schema.column_info) == 0):
        raise ValueError("Input schema is empty")

    # Try to guess which output columns we'll produce
    json_output_cols = resolve_json_output_columns(input_schema)

    json_cols = input_schema.json_columns
    column_info_objects = list(input_schema.column_info)
    if (json_cols is not None and len(json_cols) > 0):
        column_info_objects.append(
            JSONFlattenInfo(
                input_col_names=list(json_cols),
                # output_col_names=[name for name, _ in json_output_cols],
                output_col_names=json_output_cols,
                dtype="str",
                name="json_info"))

    column_info_map = {ci.name: ci for ci in column_info_objects}

    graph = build_nx_dependency_graph(column_info_objects)

    # Uncomment to print the computed: dependency layout
    # from matplotlib import pyplot as plt
    # pos = graphviz_layout(graph, prog='neato')
    # nx.draw(graph, pos, with_labels=True, font_weight='bold')
    # plt.show()

    coalesced_workflow = coalesce_ops(graph, column_info_map, preserve_re=input_schema.preserve_columns)
    if (input_schema.row_filter is not None):
        coalesced_workflow = coalesced_workflow >> Filter(f=input_schema.row_filter)

    if (visualize):
        coalesced_workflow.graph.render(view=True, format='svg')

    return nvt.Workflow(coalesced_workflow)
