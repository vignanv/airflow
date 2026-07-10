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

"""Oracle AI Vector Search hook."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

try:
    from airflow.exceptions import AirflowException
except Exception:  # pragma: no cover - useful for isolated unit testing outside Airflow
    class AirflowException(Exception):
        pass

from airflow.providers.oracle.hooks.oracle import OracleHook
from airflow.providers.oracle.vector import (
    OracleJsonFilterBuilder,
    OracleVectorDistance,
    OracleVectorFormat,
    OracleVectorIndexType,
    coerce_json_dict,
    ensure_json_serializable,
    materialize_iterable,
    normalize_distance,
    normalize_index_type,
    normalize_vector_format,
    quote_identifier,
    require_equal_lengths,
    validate_positive_int,
    vector_to_list,
)


@dataclass(frozen=True)
class OracleVectorDocument:
    """Document payload accepted by OracleVectorHook ingestion APIs."""

    id: str
    text: str
    metadata: dict[str, Any] | None = None
    embedding: Sequence[float] | None = None


@dataclass(frozen=True)
class OracleVectorSearchResult:
    """XCom-safe Oracle vector search result."""

    id: str | None
    text: str
    metadata: dict[str, Any]
    distance: float | None = None
    embedding: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class OracleVectorHook(OracleHook):
    """Hook for Oracle AI Vector Search operations.

    The hook is connection-scoped. Table names are explicit on methods so
    Airflow operators can expose them as templated task arguments.
    """

    conn_name_attr = "oracle_conn_id"
    default_conn_name = "oracle_default"
    conn_type = "oracle"
    hook_name = "Oracle Vector"

    def __init__(self, *args: Any, oracle_conn_id: str = "oracle_default", **kwargs: Any) -> None:
        super().__init__(*args, oracle_conn_id=oracle_conn_id, **kwargs)

    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------
    def get_database_version(self) -> tuple[int, ...]:
        """Return the connected Oracle database version as a tuple of ints."""
        conn = self.get_conn()
        version = getattr(conn, "version", None)
        if version is None:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version_full FROM product_component_version WHERE product LIKE 'Oracle Database%'")
                row = cursor.fetchone()
                version = row[0] if row else "0"
        return tuple(int(part) for part in str(version).split(".") if part.isdigit())

    def check_vector_support(self, *, minimum_version: tuple[int, int] = (23, 4)) -> None:
        """Raise AirflowException if the database version is too old for VECTOR support."""
        version = self.get_database_version()
        comparable = version + (0,) * max(0, len(minimum_version) - len(version))
        if comparable[: len(minimum_version)] < minimum_version:
            required = ".".join(str(part) for part in minimum_version)
            actual = ".".join(str(part) for part in version) or "unknown"
            raise AirflowException(f"Oracle AI Vector Search requires database version >= {required}; got {actual}")

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------
    def create_vector_table(
        self,
        *,
        table_name: str,
        embedding_dimension: int,
        id_column: str = "id",
        text_column: str = "text",
        metadata_column: str = "metadata",
        embedding_column: str = "embedding",
        embedding_format: OracleVectorFormat | str = OracleVectorFormat.FLOAT32,
        if_not_exists: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Create an Oracle vector table."""
        validate_positive_int("embedding_dimension", embedding_dimension)
        embedding_format = normalize_vector_format(embedding_format)
        if overwrite and if_not_exists:
            raise ValueError("overwrite=True cannot be combined with if_not_exists=True")
        if overwrite:
            self.drop_vector_table(table_name=table_name, if_exists=True)
        elif if_not_exists and self.vector_table_exists(table_name=table_name):
            self.log.info("Vector table %s already exists; skipping creation", table_name)
            return

        table_sql = quote_identifier(table_name, allow_schema=True)
        id_sql = quote_identifier(id_column)
        text_sql = quote_identifier(text_column)
        metadata_sql = quote_identifier(metadata_column)
        embedding_sql = quote_identifier(embedding_column)
        sql = f"""CREATE TABLE {table_sql} (
            {id_sql} VARCHAR2(512) PRIMARY KEY,
            {text_sql} CLOB NOT NULL,
            {metadata_sql} JSON,
            {embedding_sql} VECTOR({embedding_dimension}, {embedding_format.value}) NOT NULL
        )"""
        self.run(sql)

    def drop_vector_table(self, *, table_name: str, purge: bool = False, if_exists: bool = True) -> None:
        """Drop an Oracle vector table."""
        if if_exists and not self.vector_table_exists(table_name=table_name):
            self.log.info("Vector table %s does not exist; skipping drop", table_name)
            return
        suffix = " PURGE" if purge else ""
        self.run(f"DROP TABLE {quote_identifier(table_name, allow_schema=True)}{suffix}")

    def vector_table_exists(self, *, table_name: str) -> bool:
        """Return True when the table exists in the current schema or supplied schema."""
        owner, name = self._split_object_name(table_name)
        if owner:
            sql = "SELECT 1 FROM ALL_TABLES WHERE OWNER = :owner AND TABLE_NAME = :name"
            params = {"owner": owner.upper(), "name": name.upper()}
        else:
            sql = "SELECT 1 FROM USER_TABLES WHERE TABLE_NAME = :name"
            params = {"name": name.upper()}
        row = self.get_first(sql, parameters=params)
        return row is not None

    # ------------------------------------------------------------------
    # Ingestion and retrieval
    # ------------------------------------------------------------------
    def add_texts(
        self,
        *,
        table_name: str,
        texts: Iterable[str],
        embeddings: Iterable[Sequence[float]] | None = None,
        metadatas: Iterable[dict[str, Any] | None] | None = None,
        ids: Iterable[str] | None = None,
        id_column: str = "id",
        text_column: str = "text",
        metadata_column: str = "metadata",
        embedding_column: str = "embedding",
        batch_size: int = 1000,
        mutate_on_duplicate: bool = False,
        embedding_provider_config: dict[str, Any] | None = None,
    ) -> list[str]:
        """Insert or upsert texts and embeddings into a vector table."""
        validate_positive_int("batch_size", batch_size)
        text_list = materialize_iterable("texts", texts) or []
        embedding_list = materialize_iterable("embeddings", embeddings)
        metadata_list = materialize_iterable("metadatas", metadatas)
        id_list = materialize_iterable("ids", ids)

        if embedding_provider_config is not None:
            raise NotImplementedError(
                "DB-side embedding generation is intentionally not implemented in PR 1. "
                "Pass client-side embeddings instead."
            )
        if embedding_list is None:
            raise ValueError("embeddings is required for PR 1 client-side ingestion")

        count = require_equal_lengths(texts=text_list, embeddings=embedding_list, metadatas=metadata_list, ids=id_list)
        if id_list is None:
            id_list = [str(uuid4()) for _ in range(count)]
        if metadata_list is None:
            metadata_list = [{} for _ in range(count)]

        rows = [
            {
                "id": str(id_list[i]),
                "text": str(text_list[i]),
                "metadata": ensure_json_serializable(metadata_list[i]),
                "embedding": vector_to_list(embedding_list[i]),
            }
            for i in range(count)
        ]
        self._execute_rows(
            self._insert_or_merge_sql(
                table_name=table_name,
                id_column=id_column,
                text_column=text_column,
                metadata_column=metadata_column,
                embedding_column=embedding_column,
                mutate_on_duplicate=mutate_on_duplicate,
            ),
            rows,
            batch_size=batch_size,
        )
        return [str(item) for item in id_list]

    def add_documents(
        self,
        *,
        table_name: str,
        documents: Iterable[OracleVectorDocument | Mapping[str, Any]],
        id_column: str = "id",
        text_column: str = "text",
        metadata_column: str = "metadata",
        embedding_column: str = "embedding",
        batch_size: int = 1000,
        mutate_on_duplicate: bool = False,
        embedding_provider_config: dict[str, Any] | None = None,
    ) -> list[str]:
        """Insert or upsert structured document objects into a vector table."""
        docs = list(documents)
        texts: list[str] = []
        embeddings: list[Sequence[float] | None] = []
        metadatas: list[dict[str, Any] | None] = []
        ids: list[str] = []
        for doc in docs:
            if isinstance(doc, OracleVectorDocument):
                ids.append(doc.id)
                texts.append(doc.text)
                metadatas.append(doc.metadata)
                embeddings.append(doc.embedding)
            elif isinstance(doc, Mapping):
                ids.append(str(doc["id"]))
                texts.append(str(doc["text"]))
                metadatas.append(doc.get("metadata"))
                embeddings.append(doc.get("embedding"))
            else:
                raise ValueError("Each document must be OracleVectorDocument or a mapping")
        if any(item is None for item in embeddings) and embedding_provider_config is None:
            raise ValueError("Each document must include an embedding when embedding_provider_config is not supplied")
        return self.add_texts(
            table_name=table_name,
            texts=texts,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,
            ids=ids,
            id_column=id_column,
            text_column=text_column,
            metadata_column=metadata_column,
            embedding_column=embedding_column,
            batch_size=batch_size,
            mutate_on_duplicate=mutate_on_duplicate,
            embedding_provider_config=embedding_provider_config,
        )

    def delete(self, *, table_name: str, ids: Sequence[str], id_column: str = "id") -> int:
        """Delete documents by id and return row count."""
        if not ids:
            return 0
        table_sql = quote_identifier(table_name, allow_schema=True)
        id_sql = quote_identifier(id_column)
        rows = [{"id": str(item)} for item in ids]
        sql = f"DELETE FROM {table_sql} WHERE {id_sql} = :id"
        return self._execute_rows(sql, rows, batch_size=1000)

    def get_by_ids(
        self,
        *,
        table_name: str,
        ids: Sequence[str],
        id_column: str = "id",
        text_column: str = "text",
        metadata_column: str = "metadata",
        include_embedding: bool = False,
        embedding_column: str = "embedding",
    ) -> list[OracleVectorSearchResult]:
        """Fetch documents by id."""
        if not ids:
            return []
        table_sql = quote_identifier(table_name, allow_schema=True)
        id_sql = quote_identifier(id_column)
        text_sql = quote_identifier(text_column)
        metadata_sql = quote_identifier(metadata_column)
        embedding_sql = quote_identifier(embedding_column)
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        binds = {f"id_{i}": str(value) for i, value in enumerate(ids)}
        select_columns = [id_sql, text_sql, f"JSON_SERIALIZE({metadata_sql} RETURNING CLOB) AS metadata_json"]
        if include_embedding:
            select_columns.append(embedding_sql)
        sql = f"SELECT {', '.join(select_columns)} FROM {table_sql} WHERE {id_sql} IN ({placeholders})"
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, binds)
                rows = cursor.fetchall()
        return [self._row_to_result(row, include_score=False, include_embedding=include_embedding) for row in rows]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def similarity_search_by_vector(
        self,
        *,
        table_name: str,
        embedding: Sequence[float],
        k: int = 4,
        distance: OracleVectorDistance | str = OracleVectorDistance.EUCLIDEAN,
        filter: dict[str, Any] | None = None,
        id_column: str = "id",
        text_column: str = "text",
        metadata_column: str = "metadata",
        embedding_column: str = "embedding",
        include_score: bool = False,
        include_embedding: bool = False,
    ) -> list[OracleVectorSearchResult]:
        """Run Oracle VECTOR_DISTANCE search by query vector."""
        validate_positive_int("k", k)
        distance = normalize_distance(distance)
        table_sql = quote_identifier(table_name, allow_schema=True)
        id_sql = quote_identifier(id_column)
        text_sql = quote_identifier(text_column)
        metadata_sql = quote_identifier(metadata_column)
        embedding_sql = quote_identifier(embedding_column)
        filter_builder = OracleJsonFilterBuilder(metadata_sql)
        where_clause, filter_binds = filter_builder.build(filter)
        where_sql = f"WHERE {where_clause}" if where_clause else ""
        score_sql = f"VECTOR_DISTANCE({embedding_sql}, :query_embedding, {distance.value})"
        select_columns = [id_sql, text_sql, f"JSON_SERIALIZE({metadata_sql} RETURNING CLOB) AS metadata_json"]
        if include_score:
            select_columns.append(f"{score_sql} AS distance")
        if include_embedding:
            select_columns.append(embedding_sql)
        sql = f"""
SELECT {', '.join(select_columns)}
FROM {table_sql}
{where_sql}
ORDER BY {score_sql}
FETCH FIRST :k ROWS ONLY
"""
        binds = {"query_embedding": vector_to_list(embedding), "k": int(k), **filter_binds}
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, binds)
                rows = cursor.fetchall()
        return [
            self._row_to_result(row, include_score=include_score, include_embedding=include_embedding)
            for row in rows
        ]

    def similarity_search(
        self,
        *,
        table_name: str,
        query: str,
        embedding: Sequence[float] | None = None,
        embedding_provider_config: dict[str, Any] | None = None,
        k: int = 4,
        distance: OracleVectorDistance | str = OracleVectorDistance.EUCLIDEAN,
        filter: dict[str, Any] | None = None,
        id_column: str = "id",
        text_column: str = "text",
        metadata_column: str = "metadata",
        embedding_column: str = "embedding",
        include_score: bool = False,
        include_embedding: bool = False,
    ) -> list[OracleVectorSearchResult]:
        """Search by query text when the caller supplies the query embedding.

        PR 1 intentionally avoids owning a general embedding abstraction. The
        query text is retained for API compatibility and logging, but the caller
        must pass ``embedding``.
        """
        if embedding_provider_config is not None:
            raise NotImplementedError(
                "DB-side query embedding generation is intentionally not implemented in PR 1. "
                "Pass a client-side query embedding instead."
            )
        if embedding is None:
            raise ValueError("embedding is required for PR 1 similarity_search")
        self.log.debug("Running vector search for query text of length %s", len(query or ""))
        return self.similarity_search_by_vector(
            table_name=table_name,
            embedding=embedding,
            k=k,
            distance=distance,
            filter=filter,
            id_column=id_column,
            text_column=text_column,
            metadata_column=metadata_column,
            embedding_column=embedding_column,
            include_score=include_score,
            include_embedding=include_embedding,
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def create_vector_index(
        self,
        *,
        table_name: str,
        index_name: str,
        index_type: OracleVectorIndexType | str = OracleVectorIndexType.HNSW,
        embedding_column: str = "embedding",
        distance: OracleVectorDistance | str = OracleVectorDistance.EUCLIDEAN,
        accuracy: int | None = None,
        parallel: int | None = None,
        neighbors: int | None = None,
        ef_construction: int | None = None,
        neighbor_partitions: int | None = None,
        if_not_exists: bool = True,
    ) -> None:
        """Create an HNSW or IVF vector index."""
        if if_not_exists and self.vector_index_exists(index_name=index_name, table_name=table_name):
            self.log.info("Vector index %s already exists; skipping creation", index_name)
            return
        index_type = normalize_index_type(index_type)
        distance = normalize_distance(distance)
        validate_positive_int("accuracy", accuracy, minimum=1, maximum=100)
        validate_positive_int("parallel", parallel)
        validate_positive_int("neighbors", neighbors, minimum=2, maximum=2048)
        validate_positive_int("ef_construction", ef_construction, minimum=1, maximum=65535)
        validate_positive_int("neighbor_partitions", neighbor_partitions, minimum=1, maximum=10_000_000)
        if index_type == OracleVectorIndexType.HNSW and neighbor_partitions is not None:
            raise ValueError("neighbor_partitions is only valid for IVF indexes")
        if index_type == OracleVectorIndexType.IVF and (neighbors is not None or ef_construction is not None):
            raise ValueError("neighbors and ef_construction are only valid for HNSW indexes")

        index_sql = quote_identifier(index_name)
        table_sql = quote_identifier(table_name, allow_schema=True)
        embedding_sql = quote_identifier(embedding_column)
        parts = [f"CREATE VECTOR INDEX {index_sql} ON {table_sql} ({embedding_sql})"]
        if index_type == OracleVectorIndexType.HNSW:
            parts.append("ORGANIZATION INMEMORY NEIGHBOR GRAPH")
        else:
            parts.append("ORGANIZATION NEIGHBOR PARTITIONS")
        if accuracy is not None:
            parts.append(f"WITH TARGET ACCURACY {accuracy}")
        parts.append(f"DISTANCE {distance.value}")
        parameters: list[str] = [f"type {index_type.value}"]
        if neighbors is not None:
            parameters.append(f"neighbors {neighbors}")
        if ef_construction is not None:
            parameters.append(f"efConstruction {ef_construction}")
        if neighbor_partitions is not None:
            parameters.append(f"neighbor partitions {neighbor_partitions}")
        parts.append("PARAMETERS (" + ", ".join(parameters) + ")")
        if parallel is not None:
            parts.append(f"PARALLEL {parallel}")
        self.run("\n".join(parts))

    def drop_vector_index(self, *, index_name: str, if_exists: bool = True) -> None:
        """Drop a vector index."""
        if if_exists and not self.vector_index_exists(index_name=index_name):
            self.log.info("Vector index %s does not exist; skipping drop", index_name)
            return
        self.run(f"DROP INDEX {quote_identifier(index_name)}")

    def vector_index_exists(self, *, index_name: str, table_name: str | None = None) -> bool:
        """Return True when an index exists."""
        binds = {"index_name": index_name.upper()}
        if table_name:
            owner, table = self._split_object_name(table_name)
            if owner:
                sql = (
                    "SELECT 1 FROM ALL_INDEXES WHERE INDEX_NAME = :index_name "
                    "AND TABLE_OWNER = :owner AND TABLE_NAME = :table_name"
                )
                binds.update({"owner": owner.upper(), "table_name": table.upper()})
            else:
                sql = "SELECT 1 FROM USER_INDEXES WHERE INDEX_NAME = :index_name AND TABLE_NAME = :table_name"
                binds["table_name"] = table.upper()
        else:
            sql = "SELECT 1 FROM USER_INDEXES WHERE INDEX_NAME = :index_name"
        row = self.get_first(sql, parameters=binds)
        return row is not None

    # ------------------------------------------------------------------
    # Explicitly deferred PR 2 APIs
    # ------------------------------------------------------------------
    def max_marginal_relevance_search_by_vector(self, **_: Any) -> list[OracleVectorSearchResult]:
        raise NotImplementedError("MMR search is deferred to a follow-up PR")

    def load_onnx_model(self, **_: Any) -> None:
        raise NotImplementedError("ONNX model loading is deferred to a follow-up PR")

    def drop_onnx_model(self, **_: Any) -> None:
        raise NotImplementedError("ONNX model lifecycle support is deferred to a follow-up PR")

    def generate_embedding(self, **_: Any) -> list[float]:
        raise NotImplementedError("DB-side embedding generation is deferred to a follow-up PR")

    def generate_embeddings(self, **_: Any) -> list[list[float]]:
        raise NotImplementedError("DB-side embedding generation is deferred to a follow-up PR")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _split_object_name(self, object_name: str) -> tuple[str | None, str]:
        quote_identifier(object_name, allow_schema=True)
        parts = object_name.split(".")
        return (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])

    def _insert_or_merge_sql(
        self,
        *,
        table_name: str,
        id_column: str,
        text_column: str,
        metadata_column: str,
        embedding_column: str,
        mutate_on_duplicate: bool,
    ) -> str:
        table_sql = quote_identifier(table_name, allow_schema=True)
        id_sql = quote_identifier(id_column)
        text_sql = quote_identifier(text_column)
        metadata_sql = quote_identifier(metadata_column)
        embedding_sql = quote_identifier(embedding_column)
        if mutate_on_duplicate:
            return f"""
MERGE INTO {table_sql} tgt
USING (
    SELECT :id AS id_value, :text AS text_value, :metadata AS metadata_value, :embedding AS embedding_value
    FROM dual
) src
ON (tgt.{id_sql} = src.id_value)
WHEN MATCHED THEN UPDATE SET
    tgt.{text_sql} = src.text_value,
    tgt.{metadata_sql} = src.metadata_value,
    tgt.{embedding_sql} = src.embedding_value
WHEN NOT MATCHED THEN INSERT ({id_sql}, {text_sql}, {metadata_sql}, {embedding_sql})
VALUES (src.id_value, src.text_value, src.metadata_value, src.embedding_value)
"""
        return f"""
INSERT INTO {table_sql} ({id_sql}, {text_sql}, {metadata_sql}, {embedding_sql})
VALUES (:id, :text, :metadata, :embedding)
"""

    def _execute_rows(self, sql: str, rows: Sequence[Mapping[str, Any]], *, batch_size: int) -> int:
        if not rows:
            return 0
        total = 0
        with self.get_conn() as conn:
            with conn.cursor() as cursor:
                for start in range(0, len(rows), batch_size):
                    batch = list(rows[start : start + batch_size])
                    cursor.executemany(sql, batch)
                    if cursor.rowcount and cursor.rowcount > 0:
                        total += cursor.rowcount
                    else:
                        total += len(batch)
            conn.commit()
        return total

    def _row_to_result(
        self,
        row: Sequence[Any],
        *,
        include_score: bool,
        include_embedding: bool,
    ) -> OracleVectorSearchResult:
        idx = 0
        doc_id = None if row[idx] is None else str(row[idx])
        idx += 1
        text = row[idx].read() if hasattr(row[idx], "read") else str(row[idx])
        idx += 1
        metadata = coerce_json_dict(row[idx])
        idx += 1
        distance = None
        if include_score:
            distance = None if row[idx] is None else float(row[idx])
            idx += 1
        embedding = None
        if include_embedding:
            embedding = vector_to_list(row[idx])
        return OracleVectorSearchResult(id=doc_id, text=text, metadata=metadata, distance=distance, embedding=embedding)
