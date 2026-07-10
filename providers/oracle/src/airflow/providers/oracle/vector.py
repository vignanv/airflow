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

"""Utilities for Oracle AI Vector Search support."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]*$")


class OracleVectorDistance(str, Enum):
    """Supported Oracle VECTOR_DISTANCE metrics."""

    EUCLIDEAN = "EUCLIDEAN"
    COSINE = "COSINE"
    DOT = "DOT"


class OracleVectorIndexType(str, Enum):
    """Supported Oracle vector index organizations."""

    HNSW = "HNSW"
    IVF = "IVF"


class OracleVectorFormat(str, Enum):
    """Supported Oracle VECTOR storage formats."""

    INT8 = "INT8"
    FLOAT32 = "FLOAT32"
    FLOAT64 = "FLOAT64"
    BINARY = "BINARY"
    FLEXIBLE = "*"


def quote_identifier(identifier: str, *, allow_schema: bool = False) -> str:
    """Return a safely quoted Oracle identifier.

    The Oracle provider must never interpolate untrusted identifiers directly
    into SQL. Bind variables cannot be used for identifiers, so this helper
    restricts identifiers to ordinary Oracle identifier characters and quotes
    each component.
    """
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Identifier must be a non-empty string")
    if "\x00" in identifier:
        raise ValueError("Identifier contains an invalid null byte")

    parts = identifier.split(".")
    if len(parts) > 2 or (len(parts) == 2 and not allow_schema):
        raise ValueError(f"Invalid identifier: {identifier!r}")

    quoted: list[str] = []
    for part in parts:
        if not _IDENTIFIER_RE.fullmatch(part):
            raise ValueError(f"Unsafe Oracle identifier: {identifier!r}")
        quoted.append('"' + part.upper() + '"')
    return ".".join(quoted)


def normalize_distance(distance: OracleVectorDistance | str) -> OracleVectorDistance:
    """Normalize a user supplied distance metric."""
    if isinstance(distance, OracleVectorDistance):
        return distance
    try:
        return OracleVectorDistance(str(distance).upper())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in OracleVectorDistance)
        raise ValueError(f"Unsupported Oracle vector distance {distance!r}. Expected one of: {allowed}") from exc


def normalize_index_type(index_type: OracleVectorIndexType | str) -> OracleVectorIndexType:
    """Normalize a user supplied index type."""
    if isinstance(index_type, OracleVectorIndexType):
        return index_type
    try:
        return OracleVectorIndexType(str(index_type).upper())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in OracleVectorIndexType)
        raise ValueError(f"Unsupported Oracle vector index type {index_type!r}. Expected one of: {allowed}") from exc


def normalize_vector_format(vector_format: OracleVectorFormat | str) -> OracleVectorFormat:
    """Normalize a user supplied vector storage format."""
    if isinstance(vector_format, OracleVectorFormat):
        return vector_format
    try:
        return OracleVectorFormat(str(vector_format).upper())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in OracleVectorFormat)
        raise ValueError(f"Unsupported Oracle vector format {vector_format!r}. Expected one of: {allowed}") from exc


def vector_to_list(value: Any) -> list[float]:
    """Convert an Oracle VECTOR/python-oracledb vector-like value to list[float]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, tuple):
        return [float(v) for v in value]
    if hasattr(value, "tolist"):
        return [float(v) for v in value.tolist()]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return [float(v) for v in json.loads(stripped)]
    try:
        return [float(v) for v in value]
    except TypeError as exc:
        raise ValueError(f"Cannot convert value of type {type(value).__name__!r} to vector list") from exc


def coerce_json_dict(value: Any) -> dict[str, Any]:
    """Convert Oracle JSON/CLOB values to dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "read"):
        value = value.read()
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        if not value.strip():
            return {}
        loaded = json.loads(value)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("Metadata JSON value must be an object")
        return loaded
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"Cannot convert value of type {type(value).__name__!r} to metadata dict")


def ensure_json_serializable(value: Mapping[str, Any] | None) -> str:
    """Serialize metadata as compact JSON object text."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("Metadata must be a mapping/dictionary")
    return json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def materialize_iterable(name: str, value: Iterable[Any] | None) -> list[Any] | None:
    """Materialize an iterable once so input lengths can be validated."""
    if value is None:
        return None
    return list(value)


def require_equal_lengths(**items: Sequence[Any] | None) -> int:
    """Require all non-None sequences to have identical lengths."""
    lengths = {name: len(value) for name, value in items.items() if value is not None}
    if not lengths:
        return 0
    expected = next(iter(lengths.values()))
    mismatched = {name: length for name, length in lengths.items() if length != expected}
    if mismatched:
        detail = ", ".join(f"{name}={length}" for name, length in sorted(lengths.items()))
        raise ValueError(f"Input lengths must match: {detail}")
    return expected


def validate_positive_int(name: str, value: int | None, *, minimum: int = 1, maximum: int | None = None) -> None:
    """Validate an optional positive integer range."""
    if value is None:
        return
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


class OracleJsonFilterBuilder:
    """Translate supported JSON metadata filters into Oracle SQL predicates.

    The generated SQL always uses bind variables for values. Only JSON field
    names and operators are interpreted, and field names are strictly validated.
    """

    _COMPARISON_OPERATORS = {
        "$eq",
        "$ne",
        "$gt",
        "$lt",
        "$gte",
        "$lte",
        "$between",
        "$like",
        "$startsWith",
        "$hasSubstring",
        "$instr",
        "$in",
        "$nin",
        "$exists",
        "$not",
    }
    _LOGICAL_OPERATORS = {"$and", "$or", "$nor"}
    _FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

    def __init__(self, metadata_column_sql: str, *, bind_prefix: str = "vf") -> None:
        self.metadata_column_sql = metadata_column_sql
        self.bind_prefix = bind_prefix
        self.binds: dict[str, Any] = {}
        self._counter = 0

    def build(self, filter_: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
        if not filter_:
            return "", {}
        clause = self._parse_mapping(filter_)
        return clause, dict(self.binds)

    def _new_bind(self, value: Any) -> str:
        self._counter += 1
        name = f"{self.bind_prefix}_{self._counter}"
        self.binds[name] = value
        return f":{name}"

    def _json_path(self, field: str) -> str:
        if not self._FIELD_RE.fullmatch(field):
            raise ValueError(f"Invalid metadata filter field: {field!r}")
        return "$." + ".".join(field.split("."))

    def _json_value(self, field: str) -> str:
        path = self._json_path(field).replace("'", "''")
        return f"JSON_VALUE({self.metadata_column_sql}, '{path}' RETURNING VARCHAR2(4000) NULL ON ERROR)"

    def _json_exists(self, field: str) -> str:
        path = self._json_path(field).replace("'", "''")
        return f"JSON_EXISTS({self.metadata_column_sql}, '{path}')"

    def _parse_mapping(self, mapping: Mapping[str, Any]) -> str:
        if not isinstance(mapping, Mapping):
            raise ValueError("Metadata filter must be a mapping")
        parts: list[str] = []
        for key, value in mapping.items():
            if key in self._LOGICAL_OPERATORS:
                parts.append(self._parse_logical(key, value))
            elif key.startswith("$"):
                raise ValueError(f"Unsupported logical metadata filter operator: {key}")
            else:
                parts.append(self._parse_field(key, value))
        if not parts:
            return "1 = 1"
        return "(" + " AND ".join(parts) + ")"

    def _parse_logical(self, operator: str, value: Any) -> str:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"{operator} requires a list of filter expressions")
        expressions = [self._parse_mapping(item) for item in value]
        if not expressions:
            raise ValueError(f"{operator} requires at least one filter expression")
        if operator == "$and":
            return "(" + " AND ".join(expressions) + ")"
        if operator == "$or":
            return "(" + " OR ".join(expressions) + ")"
        if operator == "$nor":
            return "NOT (" + " OR ".join(expressions) + ")"
        raise ValueError(f"Unsupported logical operator: {operator}")

    def _parse_field(self, field: str, value: Any) -> str:
        if isinstance(value, Mapping) and any(str(k).startswith("$") for k in value):
            parts = [self._parse_comparison(field, str(op), operand) for op, operand in value.items()]
            return "(" + " AND ".join(parts) + ")"
        return self._parse_comparison(field, "$eq", value)

    def _parse_comparison(self, field: str, operator: str, operand: Any) -> str:
        if operator not in self._COMPARISON_OPERATORS:
            raise ValueError(f"Unsupported metadata filter operator: {operator}")
        json_value = self._json_value(field)

        if operator == "$exists":
            return self._json_exists(field) if bool(operand) else f"NOT {self._json_exists(field)}"
        if operator == "$not":
            return f"NOT ({self._parse_field(field, operand)})"
        if operator == "$eq":
            return f"{json_value} = {self._new_bind(operand)}"
        if operator == "$ne":
            return f"({json_value} <> {self._new_bind(operand)} OR {json_value} IS NULL)"
        if operator == "$gt":
            return f"TO_NUMBER({json_value} DEFAULT NULL ON CONVERSION ERROR) > {self._new_bind(operand)}"
        if operator == "$lt":
            return f"TO_NUMBER({json_value} DEFAULT NULL ON CONVERSION ERROR) < {self._new_bind(operand)}"
        if operator == "$gte":
            return f"TO_NUMBER({json_value} DEFAULT NULL ON CONVERSION ERROR) >= {self._new_bind(operand)}"
        if operator == "$lte":
            return f"TO_NUMBER({json_value} DEFAULT NULL ON CONVERSION ERROR) <= {self._new_bind(operand)}"
        if operator == "$between":
            if not isinstance(operand, Sequence) or isinstance(operand, (str, bytes, bytearray)) or len(operand) != 2:
                raise ValueError("$between requires a two-item sequence")
            return (
                f"TO_NUMBER({json_value} DEFAULT NULL ON CONVERSION ERROR) BETWEEN "
                f"{self._new_bind(operand[0])} AND {self._new_bind(operand[1])}"
            )
        if operator == "$like":
            return f"{json_value} LIKE {self._new_bind(operand)}"
        if operator == "$startsWith":
            return f"{json_value} LIKE {self._new_bind(str(operand) + '%')}"
        if operator == "$hasSubstring":
            return f"{json_value} LIKE {self._new_bind('%' + str(operand) + '%')}"
        if operator == "$instr":
            return f"INSTR({json_value}, {self._new_bind(operand)}) > 0"
        if operator in {"$in", "$nin"}:
            if not isinstance(operand, Sequence) or isinstance(operand, (str, bytes, bytearray)):
                raise ValueError(f"{operator} requires a list of values")
            if len(operand) == 0:
                return "1 = 0" if operator == "$in" else "1 = 1"
            placeholders = ", ".join(self._new_bind(item) for item in operand)
            comparison = f"{json_value} IN ({placeholders})"
            return comparison if operator == "$in" else f"NOT ({comparison})"
        raise ValueError(f"Unsupported metadata filter operator: {operator}")
