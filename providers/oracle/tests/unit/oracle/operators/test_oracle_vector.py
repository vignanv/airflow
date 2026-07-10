# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

from unittest import mock

import pytest

from airflow.providers.oracle.hooks.oracle_vector import OracleVectorSearchResult
from airflow.providers.oracle.operators.oracle_vector import (
    OracleAddVectorDocumentsOperator,
    OracleCreateVectorIndexOperator,
    OracleCreateVectorTableOperator,
    OracleDeleteVectorDocumentsOperator,
    OracleVectorSearchOperator,
)
from airflow.providers.oracle.vector import OracleVectorFormat


@mock.patch("airflow.providers.oracle.operators.oracle_vector.OracleVectorHook")
def test_create_vector_table_operator_calls_hook(mock_hook_class):
    op = OracleCreateVectorTableOperator(
        task_id="t",
        table_name="docs",
        embedding_dimension=3,
        embedding_format=OracleVectorFormat.INT8,
    )
    op.execute({})
    mock_hook_class.return_value.create_vector_table.assert_called_once_with(
        table_name="docs",
        embedding_dimension=3,
        id_column="id",
        text_column="text",
        metadata_column="metadata",
        embedding_column="embedding",
        embedding_format=OracleVectorFormat.INT8,
        if_not_exists=True,
        overwrite=False,
    )


@mock.patch("airflow.providers.oracle.operators.oracle_vector.OracleVectorHook")
def test_add_documents_operator_calls_hook(mock_hook_class):
    mock_hook_class.return_value.add_documents.return_value = ["d1"]
    op = OracleAddVectorDocumentsOperator(
        task_id="t",
        table_name="docs",
        documents=[{"id": "d1", "text": "hello", "embedding": [1, 2, 3]}],
    )
    assert op.execute({}) == ["d1"]
    mock_hook_class.return_value.add_documents.assert_called_once()


def test_add_documents_operator_requires_source():
    with pytest.raises(ValueError):
        OracleAddVectorDocumentsOperator(task_id="t", table_name="docs")


@mock.patch("airflow.providers.oracle.operators.oracle_vector.OracleVectorHook")
def test_search_operator_returns_serializable_results(mock_hook_class):
    mock_hook_class.return_value.similarity_search_by_vector.return_value = [
        OracleVectorSearchResult(id="d1", text="hello", metadata={"source": "unit"}, distance=0.1)
    ]
    op = OracleVectorSearchOperator(task_id="t", table_name="docs", embedding=[1, 2, 3])
    assert op.execute({}) == [
        {"id": "d1", "text": "hello", "metadata": {"source": "unit"}, "distance": 0.1, "embedding": None}
    ]


@mock.patch("airflow.providers.oracle.operators.oracle_vector.OracleVectorHook")
def test_create_index_operator_calls_hook(mock_hook_class):
    op = OracleCreateVectorIndexOperator(
        task_id="t",
        table_name="docs",
        index_name="docs_idx",
        index_type="HNSW",
        distance="COSINE",
    )
    op.execute({})
    mock_hook_class.return_value.create_vector_index.assert_called_once()


@mock.patch("airflow.providers.oracle.operators.oracle_vector.OracleVectorHook")
def test_delete_operator_calls_hook(mock_hook_class):
    mock_hook_class.return_value.delete.return_value = 2
    op = OracleDeleteVectorDocumentsOperator(task_id="t", table_name="docs", ids=["d1", "d2"])
    assert op.execute({}) == 2
    mock_hook_class.return_value.delete.assert_called_once_with(table_name="docs", ids=["d1", "d2"], id_column="id")
