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

from airflow.providers.oracle.hooks.oracle_vector import OracleVectorDocument, OracleVectorHook
from airflow.providers.oracle.utils.vector import OracleJsonFilterBuilder, quote_identifier


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.sql = None
        self.binds = None
        self.executed_batches = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, binds=None):
        self.sql = sql
        self.binds = binds or {}

    def executemany(self, sql, rows):
        self.sql = sql
        self.executed_batches.append(list(rows))
        self.rowcount = len(rows)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.version = "23.4.0.0.0"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


class RecordingOracleVectorHook(OracleVectorHook):
    def __init__(self):
        self.statements = []
        self.first_rows = []
        self.log = mock.Mock()

    def run(self, sql, *args, **kwargs):
        self.statements.append(sql)

    def get_first(self, sql, parameters=None):
        self.statements.append((sql, parameters))
        return self.first_rows.pop(0) if self.first_rows else None


def test_quote_identifier_accepts_safe_names():
    assert quote_identifier("docs") == '"DOCS"'
    assert quote_identifier("my_schema.docs", allow_schema=True) == '"MY_SCHEMA"."DOCS"'


@pytest.mark.parametrize("identifier", ["docs;drop table x", "docs where 1=1", "1docs", "docs--", "schema.table.extra"])
def test_quote_identifier_rejects_unsafe_names(identifier):
    with pytest.raises(ValueError):
        quote_identifier(identifier, allow_schema=True)


def test_create_vector_table_emits_expected_ddl():
    hook = RecordingOracleVectorHook()
    hook.create_vector_table(table_name="docs", embedding_dimension=3, if_not_exists=False)
    sql = hook.statements[-1]
    assert "CREATE TABLE" in sql
    assert '"DOCS"' in sql
    assert '"EMBEDDING" VECTOR(3, FLOAT32) NOT NULL' in sql
    assert '"METADATA" JSON' in sql


def test_create_vector_table_rejects_overwrite_and_if_not_exists():
    hook = RecordingOracleVectorHook()
    with pytest.raises(ValueError):
        hook.create_vector_table(table_name="docs", embedding_dimension=3, overwrite=True, if_not_exists=True)


def test_add_texts_validates_lengths():
    hook = RecordingOracleVectorHook()
    with pytest.raises(ValueError):
        hook.add_texts(table_name="docs", texts=["a", "b"], embeddings=[[1, 2, 3]])


def test_add_documents_executes_insert_and_commits(monkeypatch):
    cursor = FakeCursor()
    conn = FakeConnection(cursor)
    hook = RecordingOracleVectorHook()
    monkeypatch.setattr(hook, "get_conn", lambda: conn)
    ids = hook.add_documents(
        table_name="docs",
        documents=[OracleVectorDocument(id="d1", text="hello", metadata={"source": "unit"}, embedding=[1, 2, 3])],
    )
    assert ids == ["d1"]
    assert "INSERT INTO" in cursor.sql
    assert cursor.executed_batches[0][0]["id"] == "d1"
    assert cursor.executed_batches[0][0]["embedding"] == [1.0, 2.0, 3.0]
    assert conn.committed


def test_add_texts_mutate_on_duplicate_uses_merge(monkeypatch):
    cursor = FakeCursor()
    conn = FakeConnection(cursor)
    hook = RecordingOracleVectorHook()
    monkeypatch.setattr(hook, "get_conn", lambda: conn)
    hook.add_texts(
        table_name="docs",
        texts=["hello"],
        embeddings=[[1, 2, 3]],
        ids=["d1"],
        mutate_on_duplicate=True,
    )
    assert "MERGE INTO" in cursor.sql
    assert "WHEN MATCHED THEN UPDATE" in cursor.sql


def test_similarity_search_by_vector_emits_vector_distance_and_binds(monkeypatch):
    cursor = FakeCursor(rows=[("d1", "hello", '{"source":"unit"}', 0.1)])
    conn = FakeConnection(cursor)
    hook = RecordingOracleVectorHook()
    monkeypatch.setattr(hook, "get_conn", lambda: conn)
    results = hook.similarity_search_by_vector(
        table_name="docs",
        embedding=[1, 2, 3],
        k=1,
        distance="COSINE",
        filter={"source": {"$eq": "unit"}},
        include_score=True,
    )
    assert "VECTOR_DISTANCE" in cursor.sql
    assert "COSINE" in cursor.sql
    assert cursor.binds["query_embedding"] == [1.0, 2.0, 3.0]
    assert cursor.binds["k"] == 1
    assert results[0].id == "d1"
    assert results[0].metadata == {"source": "unit"}
    assert results[0].distance == 0.1


def test_get_by_ids_can_include_embedding(monkeypatch):
    cursor = FakeCursor(rows=[("d1", "hello", '{"source":"unit"}', [1, 2, 3])])
    conn = FakeConnection(cursor)
    hook = RecordingOracleVectorHook()
    monkeypatch.setattr(hook, "get_conn", lambda: conn)
    results = hook.get_by_ids(table_name="docs", ids=["d1"], include_embedding=True)
    assert "IN (:id_0)" in cursor.sql
    assert cursor.binds == {"id_0": "d1"}
    assert results[0].embedding == [1.0, 2.0, 3.0]


def test_delete_executes_delete(monkeypatch):
    cursor = FakeCursor()
    conn = FakeConnection(cursor)
    hook = RecordingOracleVectorHook()
    monkeypatch.setattr(hook, "get_conn", lambda: conn)
    deleted = hook.delete(table_name="docs", ids=["d1", "d2"])
    assert deleted == 2
    assert "DELETE FROM" in cursor.sql
    assert len(cursor.executed_batches[0]) == 2


def test_filter_builder_logical_operators():
    builder = OracleJsonFilterBuilder('"METADATA"')
    clause, binds = builder.build({"$and": [{"source": "unit"}, {"version": {"$gte": 2}}]})
    assert "JSON_VALUE" in clause
    assert "AND" in clause
    assert binds["vf_1"] == "unit"
    assert binds["vf_2"] == 2


def test_filter_builder_rejects_unknown_operator():
    builder = OracleJsonFilterBuilder('"METADATA"')
    with pytest.raises(ValueError):
        builder.build({"source": {"$bad": "x"}})


def test_create_hnsw_index_emits_ddl():
    hook = RecordingOracleVectorHook()
    hook.create_vector_index(
        table_name="docs",
        index_name="docs_hnsw_idx",
        index_type="HNSW",
        distance="COSINE",
        accuracy=90,
        parallel=2,
        neighbors=32,
        ef_construction=200,
    )
    sql = hook.statements[-1]
    assert "CREATE VECTOR INDEX" in sql
    assert "ORGANIZATION INMEMORY NEIGHBOR GRAPH" in sql
    assert "DISTANCE COSINE" in sql
    assert "neighbors 32" in sql
    assert "efConstruction 200" in sql


def test_create_ivf_index_emits_ddl():
    hook = RecordingOracleVectorHook()
    hook.create_vector_index(
        table_name="docs",
        index_name="docs_ivf_idx",
        index_type="IVF",
        neighbor_partitions=10,
    )
    sql = hook.statements[-1]
    assert "ORGANIZATION NEIGHBOR PARTITIONS" in sql
    assert "type IVF" in sql
    assert "neighbor partitions 10" in sql


def test_create_index_validates_parameter_combinations():
    hook = RecordingOracleVectorHook()
    with pytest.raises(ValueError):
        hook.create_vector_index(table_name="docs", index_name="idx", index_type="IVF", neighbors=10)
    with pytest.raises(ValueError):
        hook.create_vector_index(table_name="docs", index_name="idx", index_type="HNSW", neighbor_partitions=10)
