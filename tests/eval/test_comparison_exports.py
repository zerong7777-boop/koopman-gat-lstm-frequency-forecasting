from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from koopman_gat_lstm.eval.comparison import write_formal_comparison_artifacts


def _write_run_artifacts(
    run_dir: Path,
    *,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    case_ids: list[str],
    selected_case_id: str,
    layer1_entropy: np.ndarray,
    layer2_entropy: np.ndarray,
) -> None:
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "cases" / selected_case_id).mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "predictions" / "test_predictions.npz",
        y_pred=y_pred,
        y_true=y_true,
        case_ids=np.asarray(case_ids, dtype=str),
    )
    np.save(run_dir / "cases" / selected_case_id / "layer1_entropy.npy", layer1_entropy)
    np.save(run_dir / "cases" / selected_case_id / "layer2_entropy.npy", layer2_entropy)


def test_write_formal_comparison_artifacts_exports_expected_files_and_metrics(tmp_path):
    koopman_run_dir = tmp_path / "koopman"
    standard_run_dir = tmp_path / "standard"
    comparison_dir = tmp_path / "comparison"
    selected_case_id = "case-1"
    case_ids = ["case-1", "case-2"]
    y_true = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]],
            [[0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]],
        ],
        dtype=np.float32,
    )
    koopman_y_pred = y_true + np.array(
        [
            [[0.1, -0.1, 0.0, 0.05], [0.0, 0.05, -0.05, 0.0]],
            [[0.0, 0.0, 0.05, -0.05], [0.05, -0.05, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    standard_y_pred = y_true + np.array(
        [
            [[0.4, -0.3, 0.2, -0.1], [0.2, 0.3, -0.25, 0.15]],
            [[-0.2, 0.25, 0.15, -0.3], [0.35, -0.4, 0.2, -0.2]],
        ],
        dtype=np.float32,
    )
    layer1_entropy = np.array([0.5, 0.25, 0.125], dtype=np.float64)
    layer2_entropy = np.array([0.2, 0.15, 0.1], dtype=np.float64)

    _write_run_artifacts(
        koopman_run_dir,
        y_pred=koopman_y_pred,
        y_true=y_true,
        case_ids=case_ids,
        selected_case_id=selected_case_id,
        layer1_entropy=layer1_entropy,
        layer2_entropy=layer2_entropy,
    )
    _write_run_artifacts(
        standard_run_dir,
        y_pred=standard_y_pred,
        y_true=y_true,
        case_ids=case_ids,
        selected_case_id=selected_case_id,
        layer1_entropy=layer1_entropy + 0.1,
        layer2_entropy=layer2_entropy + 0.2,
    )

    result = write_formal_comparison_artifacts(
        koopman_run_dir=koopman_run_dir,
        standard_run_dir=standard_run_dir,
        comparison_dir=comparison_dir,
        selected_case_id=selected_case_id,
        key_nodes=[1, 4],
        input_start_s=5.0,
        sample_rate_hz=100,
    )

    assert result["selected_case_id"] == selected_case_id
    assert result["overall_metrics"]["koopman"]["rmse"] < result["overall_metrics"]["standard"]["rmse"]
    assert Path(result["paths"]["overall_metrics_comparison"]).is_file()
    assert Path(result["paths"]["key_node_metrics_comparison"]).is_file()
    assert Path(result["paths"]["attention_entropy_comparison"]).is_file()
    assert Path(result["paths"]["attention_entropy_comparison_layer1"]).is_file()
    assert Path(result["paths"]["attention_entropy_comparison_layer2"]).is_file()

    overall = pd.read_csv(result["paths"]["overall_metrics_comparison"])
    assert list(overall.columns) == ["model", "mse", "rmse", "mae"]
    assert set(overall["model"]) == {"koopman", "standard"}

    key_nodes = pd.read_csv(result["paths"]["key_node_metrics_comparison"])
    assert list(key_nodes.columns) == [
        "node",
        "koopman_mse",
        "standard_mse",
        "koopman_rmse",
        "standard_rmse",
        "koopman_mae",
        "standard_mae",
        "rmse_delta",
        "mae_delta",
    ]
    assert key_nodes["node"].tolist() == [1, 4]
    assert (key_nodes["rmse_delta"] < 0).all()
    assert (key_nodes["mae_delta"] < 0).all()

    expected_node1_rmse = float(np.sqrt((0.1**2 + 0.0**2 + 0.0**2 + 0.05**2) / 4.0))
    expected_node1_mae = 0.0375
    assert np.isclose(key_nodes.loc[key_nodes["node"] == 1, "koopman_rmse"].item(), expected_node1_rmse)
    assert np.isclose(key_nodes.loc[key_nodes["node"] == 1, "koopman_mae"].item(), expected_node1_mae)

    np.testing.assert_allclose(
        key_nodes["rmse_delta"],
        key_nodes["koopman_rmse"] - key_nodes["standard_rmse"],
    )
    np.testing.assert_allclose(
        key_nodes["mae_delta"],
        key_nodes["koopman_mae"] - key_nodes["standard_mae"],
    )

    entropy = pd.read_csv(result["paths"]["attention_entropy_comparison"])
    assert list(entropy.columns) == [
        "time_step",
        "time_s",
        "koopman_layer1_entropy",
        "standard_layer1_entropy",
        "koopman_layer2_entropy",
        "standard_layer2_entropy",
    ]
    assert len(entropy) == len(layer1_entropy)


def test_write_formal_comparison_artifacts_rejects_case_id_order_mismatch(tmp_path):
    koopman_run_dir = tmp_path / "koopman"
    standard_run_dir = tmp_path / "standard"
    comparison_dir = tmp_path / "comparison"
    selected_case_id = "case-1"
    y_true = np.array([[[1.0, 2.0]]], dtype=np.float32)
    _write_run_artifacts(
        koopman_run_dir,
        y_pred=y_true + 0.1,
        y_true=y_true,
        case_ids=["case-1"],
        selected_case_id=selected_case_id,
        layer1_entropy=np.array([0.1], dtype=np.float64),
        layer2_entropy=np.array([0.2], dtype=np.float64),
    )
    _write_run_artifacts(
        standard_run_dir,
        y_pred=y_true + 0.2,
        y_true=y_true,
        case_ids=["case-2"],
        selected_case_id=selected_case_id,
        layer1_entropy=np.array([0.1], dtype=np.float64),
        layer2_entropy=np.array([0.2], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="case ID order"):
        write_formal_comparison_artifacts(
            koopman_run_dir=koopman_run_dir,
            standard_run_dir=standard_run_dir,
            comparison_dir=comparison_dir,
            selected_case_id=selected_case_id,
            key_nodes=[1],
        )


def test_write_formal_comparison_artifacts_rejects_missing_selected_case(tmp_path):
    koopman_run_dir = tmp_path / "koopman"
    standard_run_dir = tmp_path / "standard"
    comparison_dir = tmp_path / "comparison"
    y_true = np.array([[[1.0, 2.0]]], dtype=np.float32)
    _write_run_artifacts(
        koopman_run_dir,
        y_pred=y_true + 0.1,
        y_true=y_true,
        case_ids=["case-1"],
        selected_case_id="case-1",
        layer1_entropy=np.array([0.1], dtype=np.float64),
        layer2_entropy=np.array([0.2], dtype=np.float64),
    )
    _write_run_artifacts(
        standard_run_dir,
        y_pred=y_true + 0.2,
        y_true=y_true,
        case_ids=["case-1"],
        selected_case_id="case-1",
        layer1_entropy=np.array([0.1], dtype=np.float64),
        layer2_entropy=np.array([0.2], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="selected case"):
        write_formal_comparison_artifacts(
            koopman_run_dir=koopman_run_dir,
            standard_run_dir=standard_run_dir,
            comparison_dir=comparison_dir,
            selected_case_id="missing-case",
            key_nodes=[1],
        )
