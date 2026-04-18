import numpy as np
import torch

from koopman_gat_lstm.data.artifacts import MaterializedDataset
from koopman_gat_lstm.data.torch_dataset import ForecastDataset
from koopman_gat_lstm.models.standard import StandardGATLSTM


def test_standard_gat_lstm_forward_returns_forecast_without_koopman():
    model = StandardGATLSTM(
        num_nodes=3,
        input_feature_dim=1,
        gat_hidden_dim=4,
        gat_heads_layer1=2,
        gat_heads_layer2=2,
        lstm_hidden_dim=5,
        forecast_steps=6,
    )
    y = model(torch.randn(2, 4, 3), torch.eye(3))

    assert y.shape == (2, 6, 3)


def test_forecast_dataset_returns_case_selected_tensors():
    materialized = MaterializedDataset(
        case_ids=np.array(["case-a", "case-b"]),
        x=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        y=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        koopman=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        adjacency=np.eye(3, dtype=np.float32),
        frequency_mean=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        frequency_std=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        split={"train": ["case-a"], "val": [], "test": ["case-b"]},
    )

    sample = ForecastDataset(materialized, ["case-b"])[0]

    assert sample["case_id"] == "case-b"
    assert torch.equal(sample["x"], torch.as_tensor(materialized.x[1], dtype=torch.float32))
    assert torch.equal(sample["y"], torch.as_tensor(materialized.y[1], dtype=torch.float32))
    assert torch.equal(sample["koopman"], torch.as_tensor(materialized.koopman[1], dtype=torch.float32))
