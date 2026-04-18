import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from koopman_gat_lstm.eval.evaluator import evaluate_model


class ToyAttentionModel(torch.nn.Module):
    def forward(self, x, adjacency, return_attention=False):
        y_pred = x[:, :1, :]
        attention = {
            "layer1": torch.full(
                (x.shape[0], x.shape[1], 1, x.shape[2], x.shape[2]),
                0.5,
                dtype=x.dtype,
                device=x.device,
            ),
            "layer2": torch.full(
                (x.shape[0], x.shape[1], 1, x.shape[2], x.shape[2]),
                0.5,
                dtype=x.dtype,
                device=x.device,
            ),
        }
        if return_attention:
            return y_pred, attention
        return y_pred


def _case(case_id, value, time_in=3, nodes=2):
    return {
        "case_id": case_id,
        "x": torch.full((time_in, nodes), value),
        "y": torch.full((1, nodes), value),
        "koopman": torch.full((nodes,), value),
    }


def _evaluate(tmp_path, dataset, **overrides):
    options = {
        "model": ToyAttentionModel(),
        "loader": DataLoader(dataset, batch_size=1),
        "adjacency": torch.eye(2),
        "run_dir": tmp_path,
        "case_ids": [row["case_id"] for row in dataset],
        "uses_koopman": False,
        "device": torch.device("cpu"),
        "selected_case_id": dataset[0]["case_id"] if dataset else None,
    }
    options.update(overrides)
    return evaluate_model(**options)


def test_evaluate_model_writes_metrics_predictions_and_selected_case_entropy(tmp_path):
    time_in = 3
    dataset = [_case("case-a", 1.0, time_in=time_in), _case("case-b", 0.0, time_in=time_in)]
    loader = DataLoader(dataset, batch_size=2)

    result = evaluate_model(
        ToyAttentionModel(),
        loader,
        torch.eye(2),
        tmp_path,
        case_ids=["case-a", "case-b"],
        uses_koopman=False,
        device=torch.device("cpu"),
        selected_case_id="case-a",
    )

    assert result["metrics"]["rmse"] == 0.0
    assert (tmp_path / "metrics" / "test_metrics.json").is_file()
    assert (tmp_path / "predictions" / "test_predictions.npz").is_file()
    assert (tmp_path / "cases" / "case-a" / "layer1_entropy.npy").is_file()
    assert (tmp_path / "cases" / "case-a" / "layer2_entropy.npy").is_file()
    assert np.load(tmp_path / "cases" / "case-a" / "layer1_entropy.npy").shape == (time_in,)

    predictions = np.load(tmp_path / "predictions" / "test_predictions.npz")
    np.testing.assert_allclose(predictions["y_pred"], np.array([[[1.0, 1.0]], [[0.0, 0.0]]], dtype=np.float32))
    np.testing.assert_allclose(predictions["y_true"], np.array([[[1.0, 1.0]], [[0.0, 0.0]]], dtype=np.float32))
    assert predictions["case_ids"].tolist() == ["case-a", "case-b"]


def test_evaluate_model_exports_entropy_for_selected_case_in_later_batch(tmp_path):
    dataset = [_case("case-a", 1.0), _case("case-b", 0.0)]

    result = _evaluate(
        tmp_path,
        dataset,
        selected_case_id="case-b",
    )

    assert result["selected_case_id"] == "case-b"
    assert (tmp_path / "cases" / "case-b" / "layer1_entropy.npy").is_file()
    assert not (tmp_path / "cases" / "case-a" / "layer1_entropy.npy").exists()


def test_evaluate_model_rejects_case_id_count_mismatch_from_case_ids_argument(tmp_path):
    loader = [(torch.ones(2, 3, 2), torch.ones(2, 1, 2))]

    with pytest.raises(ValueError, match="case ID count"):
        evaluate_model(
            ToyAttentionModel(),
            loader,
            torch.eye(2),
            tmp_path,
            case_ids=["case-a"],
            uses_koopman=False,
            device=torch.device("cpu"),
        )


def test_evaluate_model_rejects_case_id_count_mismatch_from_batch_case_id(tmp_path):
    loader = [
        {
            "case_id": "case-a",
            "x": torch.ones(2, 3, 2),
            "y": torch.ones(2, 1, 2),
            "koopman": torch.ones(2, 2),
        }
    ]

    with pytest.raises(ValueError, match="case ID count"):
        evaluate_model(
            ToyAttentionModel(),
            loader,
            torch.eye(2),
            tmp_path,
            case_ids=["case-a", "case-b"],
            uses_koopman=False,
            device=torch.device("cpu"),
        )


def test_evaluate_model_accepts_numpy_adjacency(tmp_path):
    dataset = [_case("case-a", 1.0)]

    result = _evaluate(
        tmp_path,
        dataset,
        adjacency=np.eye(2, dtype=np.float64),
    )

    assert result["metrics"]["rmse"] == 0.0


def test_evaluate_model_rejects_unknown_selected_case(tmp_path):
    with pytest.raises(ValueError, match="selected case id was not evaluated"):
        _evaluate(tmp_path, [_case("case-a", 1.0)], selected_case_id="missing-case")


def test_evaluate_model_rejects_empty_loader(tmp_path):
    with pytest.raises(ValueError, match="loader must contain at least one batch"):
        evaluate_model(
            ToyAttentionModel(),
            [],
            torch.eye(2),
            tmp_path,
            case_ids=[],
            uses_koopman=False,
            device=torch.device("cpu"),
        )
