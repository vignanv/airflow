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

import datetime

from airflow import DAG
from airflow.providers.oracle.operators.oracle_vector import (
    OracleAddVectorDocumentsOperator,
    OracleCreateVectorIndexOperator,
    OracleCreateVectorTableOperator,
    OracleDeleteVectorDocumentsOperator,
    OracleVectorSearchOperator,
)

TABLE_NAME = "AIRFLOW_VECTOR_DOCS"
INDEX_NAME = "AIRFLOW_VECTOR_DOCS_HNSW_IDX"

with DAG(
    dag_id="example_oracle_vector",
    start_date=datetime.datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["example", "oracle", "vector"],
) as dag:
    create_table = OracleCreateVectorTableOperator(
        task_id="create_vector_table",
        table_name=TABLE_NAME,
        embedding_dimension=3,
        overwrite=True,
        if_not_exists=False,
    )

    add_documents = OracleAddVectorDocumentsOperator(
        task_id="add_documents",
        table_name=TABLE_NAME,
        documents=[
            {
                "id": "doc-1",
                "text": "Oracle Database supports AI Vector Search.",
                "metadata": {"source": "example", "topic": "oracle"},
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "id": "doc-2",
                "text": "Apache Airflow orchestrates data pipelines.",
                "metadata": {"source": "example", "topic": "airflow"},
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "id": "doc-3",
                "text": "Vector search retrieves semantically similar content.",
                "metadata": {"source": "example", "topic": "search"},
                "embedding": [0.0, 0.0, 1.0],
            },
        ],
        mutate_on_duplicate=True,
    )

    create_index = OracleCreateVectorIndexOperator(
        task_id="create_vector_index",
        table_name=TABLE_NAME,
        index_name=INDEX_NAME,
        index_type="HNSW",
        distance="COSINE",
        accuracy=90,
        neighbors=32,
        ef_construction=200,
        if_not_exists=True,
    )

    search = OracleVectorSearchOperator(
        task_id="search_documents",
        table_name=TABLE_NAME,
        embedding=[1.0, 0.0, 0.0],
        k=2,
        distance="COSINE",
        filter={"source": {"$eq": "example"}},
        include_score=True,
    )

    delete_documents = OracleDeleteVectorDocumentsOperator(
        task_id="delete_documents",
        table_name=TABLE_NAME,
        ids=["doc-1", "doc-2", "doc-3"],
    )

    create_table >> add_documents >> create_index >> search >> delete_documents
