from __future__ import annotations

from collections.abc import Iterable, Mapping

from koopman_gat_lstm.data.constants import KOOPMAN_KEY_SUFFIX
from koopman_gat_lstm.data.readers import read_source_node_labels, validate_against_canonical


def _validate_source_labels(source_name: str, labels: Iterable[object], canonical: list[str]) -> dict[str, int]:
    try:
        return validate_against_canonical(labels, canonical)
    except ValueError as exc:
        raise ValueError(f"{source_name}: {exc}") from exc


def build_alignment_index(
    frequency_cases: Mapping[str, Iterable[object]],
    koopman_rows: Mapping[str, Iterable[object]],
    adjacency_labels: Iterable[object],
    canonical_node_order: Iterable[str],
):
    try:
        canonical = read_source_node_labels(canonical_node_order)
    except ValueError as exc:
        raise ValueError(f"canonical node order: {exc}") from exc
    index = []
    unmatched = []
    original_adjacency_labels = list(adjacency_labels)
    adjacency_pos = _validate_source_labels("adjacency labels", original_adjacency_labels, canonical)
    matched_koopman_keys = set()

    for case_id, freq_labels in frequency_cases.items():
        koopman_key = f"{case_id}{KOOPMAN_KEY_SUFFIX}"
        if koopman_key not in koopman_rows:
            unmatched.append({"case_id": case_id, "reason": "missing_koopman"})
            continue
        matched_koopman_keys.add(koopman_key)

        original_frequency_labels = list(freq_labels)
        original_koopman_labels = list(koopman_rows[koopman_key])
        freq_pos = _validate_source_labels(f"frequency labels for {case_id}", original_frequency_labels, canonical)
        koopman_pos = _validate_source_labels(f"koopman labels for {case_id}", original_koopman_labels, canonical)

        index.append(
            {
                "case_id": case_id,
                "koopman_key": koopman_key,
                "original_frequency_labels": original_frequency_labels,
                "original_koopman_labels": original_koopman_labels,
                "original_adjacency_labels": original_adjacency_labels,
                "frequency_reorder": [freq_pos[label] for label in canonical],
                "koopman_reorder": [koopman_pos[label] for label in canonical],
                "adjacency_reorder": [adjacency_pos[label] for label in canonical],
            }
        )

    for koopman_key in koopman_rows:
        if koopman_key not in matched_koopman_keys:
            unmatched.append({"koopman_key": koopman_key, "reason": "orphan_koopman"})

    return index, unmatched
