from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from koopman_gat_lstm.data.constants import KOOPMAN_KEY_SUFFIX
from koopman_gat_lstm.data.readers import read_source_node_labels, validate_against_canonical


FREQUENCY_TIME_COLUMN = "时间（秒）"
KOOPMAN_DATASET_COLUMN = "Dataset_Name"
ADJACENCY_SHEET_NAME = "原始邻接矩阵A"
FREQUENCY_BUS_PATTERN = re.compile(r"(?<![A-Z0-9])BUS-?0*([1-9]\d*)(?![A-Z0-9])", re.IGNORECASE)


def _canonical_labels(canonical_node_order: Iterable[str]) -> list[str]:
    canonical = read_source_node_labels(canonical_node_order)
    return canonical


def _reorder_positions(labels: Iterable[object], canonical_node_order: Iterable[str], *, context: str) -> tuple[list[str], list[int]]:
    canonical = _canonical_labels(canonical_node_order)
    try:
        positions = validate_against_canonical(labels, canonical)
    except ValueError as exc:
        raise ValueError(f"{context}: {exc}") from exc
    return canonical, [positions[label] for label in canonical]


def extract_bus_label_from_frequency_column(column: object) -> str:
    matches = {f"BUS{int(match)}" for match in FREQUENCY_BUS_PATTERN.findall(str(column))}
    if not matches:
        raise ValueError(f"frequency column does not contain a BUS label: {column!r}")
    if len(matches) > 1:
        raise ValueError(f"frequency column contains mismatched BUS labels: {column!r}")
    return matches.pop()


def _normalize_koopman_dataset_name(dataset_name: object) -> str:
    normalized = str(dataset_name)
    if normalized.endswith(KOOPMAN_KEY_SUFFIX):
        return normalized[: -len(KOOPMAN_KEY_SUFFIX)]
    return normalized


def read_frequency_workbook(path, canonical_node_order) -> dict[str, dict[str, object]]:
    workbook = pd.read_excel(path, sheet_name=None)
    cases: dict[str, dict[str, object]] = {}

    for sheet_name, frame in workbook.items():
        if FREQUENCY_TIME_COLUMN not in frame.columns:
            raise ValueError(f"{sheet_name}: missing frequency time column {FREQUENCY_TIME_COLUMN!r}")

        value_columns = [column for column in frame.columns if column != FREQUENCY_TIME_COLUMN]
        labels = [extract_bus_label_from_frequency_column(column) for column in value_columns]
        canonical, reorder = _reorder_positions(labels, canonical_node_order, context=f"frequency labels for {sheet_name}")
        ordered_columns = [value_columns[index] for index in reorder]

        cases[sheet_name] = {
            "timestamps": frame[FREQUENCY_TIME_COLUMN].to_numpy(dtype=float),
            "values": frame.loc[:, ordered_columns].to_numpy(dtype=float),
            "node_labels": canonical,
        }

    return cases


def read_koopman_workbook(path, canonical_node_order) -> dict[str, dict[str, object]]:
    frame = pd.read_excel(path)
    if KOOPMAN_DATASET_COLUMN not in frame.columns:
        raise ValueError(f"missing Koopman column {KOOPMAN_DATASET_COLUMN!r}")

    value_columns = [column for column in frame.columns if column != KOOPMAN_DATASET_COLUMN]
    canonical, reorder = _reorder_positions(value_columns, canonical_node_order, context="koopman labels")
    ordered_columns = [value_columns[index] for index in reorder]
    rows: dict[str, dict[str, object]] = {}

    for _, row in frame.iterrows():
        dataset_name = _normalize_koopman_dataset_name(row[KOOPMAN_DATASET_COLUMN])
        if dataset_name in rows:
            raise ValueError(f"duplicate Koopman dataset name: {dataset_name}")
        rows[dataset_name] = {
            "values": row.loc[ordered_columns].to_numpy(dtype=float),
            "node_labels": canonical,
        }

    return rows


def read_adjacency_workbook(path, canonical_node_order) -> tuple[np.ndarray, list[str]]:
    try:
        frame = pd.read_excel(path, sheet_name=ADJACENCY_SHEET_NAME, index_col=0)
    except ValueError as exc:
        raise ValueError(f"missing adjacency sheet {ADJACENCY_SHEET_NAME!r}") from exc

    canonical, row_reorder = _reorder_positions(frame.index, canonical_node_order, context="adjacency row labels")
    _, column_reorder = _reorder_positions(frame.columns, canonical_node_order, context="adjacency column labels")
    matrix = frame.iloc[row_reorder, column_reorder].to_numpy(dtype=float)

    return matrix, canonical
