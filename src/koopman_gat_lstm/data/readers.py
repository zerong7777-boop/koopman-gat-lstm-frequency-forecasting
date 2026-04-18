from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real
import re


BUS_LABEL_PATTERN = re.compile(r"^BUS0*([1-9]\d*)$")
BUS_LABEL_WITH_SPACE_PATTERN = re.compile(r"^BUS\s+0*([1-9]\d*)$")
NUMERIC_LABEL_PATTERN = re.compile(r"^0*([1-9]\d*)$")


def _coerce_scalar_value(raw_label: object) -> object:
    item = getattr(raw_label, "item", None)
    if callable(item):
        coerced = item()
        if coerced is raw_label:
            return raw_label
        return _coerce_scalar_value(coerced)
    return raw_label


def normalize_node_label(raw_label: object) -> str:
    raw_label = _coerce_scalar_value(raw_label)

    if isinstance(raw_label, bool):
        raise ValueError(f"ambiguous node label: {raw_label!r}")

    if isinstance(raw_label, Integral):
        node_id = int(raw_label)
    elif isinstance(raw_label, Real):
        if not float(raw_label).is_integer():
            raise ValueError(f"ambiguous node label: {raw_label!r}")
        node_id = int(raw_label)
    elif isinstance(raw_label, str):
        stripped = raw_label.strip()
        candidate = stripped.upper()
        if not candidate:
            raise ValueError(f"ambiguous node label: {raw_label!r}")
        if any(separator in candidate for separator in ("/", ",", ";")):
            raise ValueError(f"ambiguous node label: {raw_label!r}")
        if any(char.isspace() for char in candidate):
            spaced_bus_match = BUS_LABEL_WITH_SPACE_PATTERN.fullmatch(candidate)
            if not spaced_bus_match:
                raise ValueError(f"ambiguous node label: {raw_label!r}")
            node_id = int(spaced_bus_match.group(1))
            return f"BUS{node_id}"

        bus_match = BUS_LABEL_PATTERN.fullmatch(candidate)
        numeric_match = NUMERIC_LABEL_PATTERN.fullmatch(candidate)
        if bus_match:
            node_id = int(bus_match.group(1))
        elif numeric_match:
            node_id = int(numeric_match.group(1))
        else:
            raise ValueError(f"ambiguous node label: {raw_label!r}")
    else:
        raise ValueError(f"ambiguous node label: {raw_label!r}")

    if node_id <= 0:
        raise ValueError(f"ambiguous node label: {raw_label!r}")

    return f"BUS{node_id}"


def read_source_node_labels(raw_labels: Iterable[object]) -> list[str]:
    normalized_labels = [normalize_node_label(raw_label) for raw_label in raw_labels]
    build_label_positions(normalized_labels)
    return normalized_labels


def build_label_positions(labels: Iterable[str]) -> dict[str, int]:
    positions: dict[str, int] = {}

    for index, label in enumerate(labels):
        if label in positions:
            raise ValueError(f"duplicate node label: {label}")
        positions[label] = index

    return positions


def validate_against_canonical(labels: Iterable[object], canonical_node_order: Iterable[str]) -> dict[str, int]:
    normalized_labels = read_source_node_labels(labels)
    positions = build_label_positions(normalized_labels)
    canonical = list(canonical_node_order)

    if set(positions) != set(canonical):
        raise ValueError("node labels do not match canonical order")

    return positions
