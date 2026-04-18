import pytest
import torch

from koopman_gat_lstm.eval.metrics import compute_metrics
from koopman_gat_lstm.train.trainer import TrainerDefaults, select_best_checkpoint_metric


def test_compute_metrics_returns_expected_regression_metrics():
    y_pred = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    y_true = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)

    metrics = compute_metrics(y_pred, y_true)

    assert metrics["mse"] == pytest.approx(13.0 / 3.0)
    assert metrics["rmse"] == pytest.approx((13.0 / 3.0) ** 0.5)
    assert metrics["mae"] == pytest.approx(5.0 / 3.0)


def test_trainer_defaults_use_mse_loss_and_val_rmse_checkpoint_metric():
    defaults = TrainerDefaults()

    assert isinstance(defaults.loss_fn, torch.nn.MSELoss)
    assert defaults.best_checkpoint_metric_name == "rmse"
    assert defaults.best_checkpoint_metric({"rmse": 0.125, "mae": 0.5}) == pytest.approx(0.125)


def test_select_best_checkpoint_metric_requires_rmse_by_default():
    with pytest.raises(KeyError, match="rmse"):
        select_best_checkpoint_metric({"mse": 0.25, "mae": 0.5})


def test_compute_metrics_rejects_broadcastable_shape_mismatch():
    y_pred = torch.ones(2, 1)
    y_true = torch.ones(2)

    with pytest.raises(ValueError, match="identical shapes"):
        compute_metrics(y_pred, y_true)
