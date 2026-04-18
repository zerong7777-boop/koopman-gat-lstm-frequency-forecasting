import pytest

from koopman_gat_lstm.data.alignment import build_alignment_index
from koopman_gat_lstm.data.constants import CANONICAL_BUS_ORDER, KOOPMAN_KEY_SUFFIX
from koopman_gat_lstm.data.readers import read_source_node_labels


class FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_read_source_node_labels_normalizes_workbook_style_values():
    labels = read_source_node_labels([" bus 1 ", 2, "03"])

    assert labels == ["BUS1", "BUS2", "BUS3"]


def test_read_source_node_labels_accepts_numpy_style_scalar_values():
    labels = read_source_node_labels([FakeScalar(1), FakeScalar(2.0), "BUS3"])

    assert labels == ["BUS1", "BUS2", "BUS3"]


def test_read_source_node_labels_raises_on_duplicate_normalized_labels():
    with pytest.raises(ValueError, match="duplicate node label: BUS1"):
        read_source_node_labels(["BUS1", "1"])


def test_read_source_node_labels_raises_on_ambiguous_or_malformed_labels():
    with pytest.raises(ValueError, match="ambiguous node label"):
        read_source_node_labels(["BUS1/BUS2"])


def test_read_source_node_labels_raises_on_malformed_internal_whitespace():
    with pytest.raises(ValueError, match="ambiguous node label"):
        read_source_node_labels(["3 9"])

    with pytest.raises(ValueError, match="ambiguous node label"):
        read_source_node_labels(["B US1"])


def test_build_alignment_index_records_original_labels_and_adjacency_reorder():
    frequency_cases = {
        "case-39-trip-0.500": [" BUS2 ", "bus1", "3"],
    }
    koopman_rows = {
        f"case-39-trip-0.500{KOOPMAN_KEY_SUFFIX}": ["BUS3", "BUS1", "BUS2"],
    }
    adjacency_labels = ["BUS2", "BUS3", "BUS1"]
    canonical = ["BUS1", "BUS2", "BUS3"]

    index, unmatched = build_alignment_index(
        frequency_cases=frequency_cases,
        koopman_rows=koopman_rows,
        adjacency_labels=adjacency_labels,
        canonical_node_order=canonical,
    )

    assert unmatched == []
    assert index[0]["case_id"] == "case-39-trip-0.500"
    assert index[0]["original_frequency_labels"] == [" BUS2 ", "bus1", "3"]
    assert index[0]["original_koopman_labels"] == ["BUS3", "BUS1", "BUS2"]
    assert index[0]["original_adjacency_labels"] == ["BUS2", "BUS3", "BUS1"]
    assert index[0]["frequency_reorder"] == [1, 0, 2]
    assert index[0]["koopman_reorder"] == [1, 2, 0]
    assert index[0]["adjacency_reorder"] == [2, 0, 1]


def test_build_alignment_index_raises_on_duplicate_labels():
    with pytest.raises(ValueError, match="duplicate node label: BUS1"):
        build_alignment_index(
            frequency_cases={"case": ["BUS1", "1", "BUS3"]},
            koopman_rows={f"case{KOOPMAN_KEY_SUFFIX}": ["BUS1", "BUS2", "BUS3"]},
            adjacency_labels=["BUS1", "BUS2", "BUS3"],
            canonical_node_order=["BUS1", "BUS2", "BUS3"],
        )


def test_build_alignment_index_raises_on_ambiguous_labels():
    with pytest.raises(ValueError, match="frequency labels for case: ambiguous node label"):
        build_alignment_index(
            frequency_cases={"case": ["BUS1/BUS2", "BUS2", "BUS3"]},
            koopman_rows={f"case{KOOPMAN_KEY_SUFFIX}": ["BUS1", "BUS2", "BUS3"]},
            adjacency_labels=["BUS1", "BUS2", "BUS3"],
            canonical_node_order=["BUS1", "BUS2", "BUS3"],
        )


def test_build_alignment_index_raises_on_missing_or_extra_frequency_labels():
    with pytest.raises(ValueError, match="frequency labels for case: node labels do not match canonical order"):
        build_alignment_index(
            frequency_cases={"case": ["BUS1", "BUS2", "BUS4"]},
            koopman_rows={f"case{KOOPMAN_KEY_SUFFIX}": ["BUS1", "BUS2", "BUS3"]},
            adjacency_labels=["BUS1", "BUS2", "BUS3"],
            canonical_node_order=["BUS1", "BUS2", "BUS3"],
        )


def test_build_alignment_index_raises_on_bad_adjacency_labels():
    with pytest.raises(ValueError, match="adjacency labels: ambiguous node label"):
        build_alignment_index(
            frequency_cases={"case": ["BUS1", "BUS2", "BUS3"]},
            koopman_rows={f"case{KOOPMAN_KEY_SUFFIX}": ["BUS1", "BUS2", "BUS3"]},
            adjacency_labels=["BUS1/BUS2", "BUS2", "BUS3"],
            canonical_node_order=["BUS1", "BUS2", "BUS3"],
        )


def test_build_alignment_index_raises_on_bad_koopman_labels():
    with pytest.raises(ValueError, match="koopman labels for case: ambiguous node label"):
        build_alignment_index(
            frequency_cases={"case": ["BUS1", "BUS2", "BUS3"]},
            koopman_rows={f"case{KOOPMAN_KEY_SUFFIX}": ["BUS1", "BUS2/BUS3", "BUS3"]},
            adjacency_labels=["BUS1", "BUS2", "BUS3"],
            canonical_node_order=["BUS1", "BUS2", "BUS3"],
        )


def test_build_alignment_index_raises_on_duplicate_canonical_order_labels():
    with pytest.raises(ValueError, match="canonical node order: duplicate node label: BUS1"):
        build_alignment_index(
            frequency_cases={"case": ["BUS1", "BUS2", "BUS3"]},
            koopman_rows={f"case{KOOPMAN_KEY_SUFFIX}": ["BUS1", "BUS2", "BUS3"]},
            adjacency_labels=["BUS1", "BUS2", "BUS3"],
            canonical_node_order=["BUS1", "BUS1", "BUS3"],
        )


def test_build_alignment_index_aligns_shuffled_39_node_sources_to_canonical_order():
    canonical = CANONICAL_BUS_ORDER
    frequency_labels = list(reversed(canonical))
    koopman_labels = canonical[1::2] + canonical[0::2]
    adjacency_labels = canonical[9:] + canonical[:9]

    index, unmatched = build_alignment_index(
        frequency_cases={"case39": frequency_labels},
        koopman_rows={f"case39{KOOPMAN_KEY_SUFFIX}": koopman_labels},
        adjacency_labels=adjacency_labels,
        canonical_node_order=canonical,
    )

    assert unmatched == []
    assert index[0]["frequency_reorder"] == list(range(38, -1, -1))
    assert index[0]["koopman_reorder"] == [koopman_labels.index(label) for label in canonical]
    assert index[0]["adjacency_reorder"] == [adjacency_labels.index(label) for label in canonical]


def test_build_alignment_index_reports_orphan_koopman_rows_as_unmatched():
    canonical = ["BUS1", "BUS2", "BUS3"]

    index, unmatched = build_alignment_index(
        frequency_cases={"case": canonical},
        koopman_rows={
            f"case{KOOPMAN_KEY_SUFFIX}": canonical,
            f"orphan{KOOPMAN_KEY_SUFFIX}": canonical,
        },
        adjacency_labels=canonical,
        canonical_node_order=canonical,
    )

    assert len(index) == 1
    assert unmatched == [{"koopman_key": f"orphan{KOOPMAN_KEY_SUFFIX}", "reason": "orphan_koopman"}]
