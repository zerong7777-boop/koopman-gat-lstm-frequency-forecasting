from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from koopman_gat_lstm.exports.case_exports import build_case_dir
from koopman_gat_lstm.exports.entropy_exports import compute_layer_entropy_curve


def test_compute_layer_entropy_curve_averages_nodes_then_heads():
    attention = np.array(
        [
            [
                [[0.5, 0.5], [0.25, 0.75]],
                [[0.5, 0.5], [0.5, 0.5]],
            ]
        ],
        dtype=np.float64,
    )  # time, heads, nodes, neighbors

    curve = compute_layer_entropy_curve(attention)

    first_head = (np.log(2.0) + (-(0.25 * np.log(0.25) + 0.75 * np.log(0.75)))) / 2.0
    second_head = np.log(2.0)
    expected = np.array([(first_head + second_head) / 2.0])

    assert curve.shape == (1,)
    assert np.allclose(curve, expected)


def test_compute_layer_entropy_curve_treats_zero_attention_rows_as_zero_entropy():
    attention = np.array(
        [
            [
                [[0.0, 0.0], [0.5, 0.5]],
            ]
        ],
        dtype=np.float64,
    )

    curve = compute_layer_entropy_curve(attention)

    expected = np.array([np.log(2.0) / 2.0])
    assert np.allclose(curve, expected)


def test_compute_layer_entropy_curve_averages_batch_to_global_time_curve():
    attention = np.array(
        [
            [
                [
                    [[0.5, 0.5], [0.5, 0.5]],
                ],
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                ],
            ],
            [
                [
                    [[0.5, 0.5], [0.5, 0.5]],
                ],
                [
                    [[0.5, 0.5], [0.5, 0.5]],
                ],
            ],
        ],
        dtype=np.float64,
    )  # batch, time, heads, nodes, neighbors

    curve = compute_layer_entropy_curve(attention)

    expected = np.array([np.log(2.0), np.log(2.0) / 2.0])
    assert curve.shape == (2,)
    assert np.allclose(curve, expected)


def test_build_case_dir_creates_case_subdirectory():
    run_dir = Path(__file__).resolve().parent / ".artifacts" / f"task7-run-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)

    case_dir = build_case_dir(run_dir, "case-alpha")

    assert case_dir == run_dir / "cases" / "case-alpha"
    assert case_dir.is_dir()


def test_build_case_dir_rejects_path_traversal():
    with pytest.raises(ValueError, match="leaf directory name|path separators"):
        build_case_dir(Path("run-root"), "../../escaped")


def test_build_case_dir_rejects_drive_qualified_names():
    with pytest.raises(ValueError, match="path separators or drive qualifiers"):
        build_case_dir(Path("run-root"), "C:evil")


@pytest.mark.parametrize("case_id", ["..\\..\\escaped", "foo\\bar"])
def test_build_case_dir_rejects_windows_path_separators(case_id: str):
    with pytest.raises(ValueError, match="path separators or drive qualifiers"):
        build_case_dir(Path("run-root"), case_id)


@pytest.mark.parametrize("case_id", ["/", "\\", ":"])
def test_build_case_dir_rejects_single_separator_tokens(case_id: str):
    with pytest.raises(ValueError, match="path separators or drive qualifiers"):
        build_case_dir(Path("run-root"), case_id)


@pytest.mark.parametrize("case_id", ["", ".", ".."])
def test_build_case_dir_rejects_non_leaf_case_ids(case_id: str):
    with pytest.raises(ValueError, match="leaf directory name"):
        build_case_dir(Path("run-root"), case_id)
