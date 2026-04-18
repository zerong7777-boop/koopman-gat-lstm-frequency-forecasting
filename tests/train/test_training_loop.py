import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from koopman_gat_lstm.train.trainer import train_model


class TinyKoopmanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x, adjacency, koopman):
        del adjacency
        return self.linear(x + koopman)


class MismatchedShapeModel(torch.nn.Module):
    def __init__(self, matching_calls=0):
        super().__init__()
        self.matching_calls = matching_calls
        self.calls = 0
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, adjacency, koopman=None):
        del adjacency, koopman
        self.calls += 1
        if self.calls <= self.matching_calls:
            return x.sum(dim=-1, keepdim=True) * self.scale
        return x * self.scale


def _dataset():
    x = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [2.0, 0.0]],
            [[2.0, 1.0], [3.0, 0.0]],
            [[3.0, 1.0], [4.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    koopman = torch.zeros_like(x)
    y = x.sum(dim=-1, keepdim=True)
    return TensorDataset(x, y, koopman)


@pytest.fixture
def adjacency():
    return torch.eye(2, dtype=torch.float32)


def test_train_model_saves_best_checkpoint_and_returns_history(tmp_path):
    dataset = _dataset()
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)
    adjacency = torch.eye(2, dtype=torch.float32)

    result = train_model(
        model=TinyKoopmanModel(),
        train_loader=train_loader,
        val_loader=val_loader,
        adjacency=adjacency,
        run_dir=tmp_path,
        max_epochs=3,
        learning_rate=0.01,
        patience=3,
        device="cpu",
        uses_koopman=True,
    )

    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    assert checkpoint_path.is_file()
    assert result["best_val_rmse"] >= 0
    assert len(result["history"]) >= 1
    assert result["best_val_rmse"] == min(
        epoch["val_rmse"] for epoch in result["history"]
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert {"model_state_dict", "optimizer_state_dict", "epoch", "val_rmse"} <= set(
        checkpoint
    )


def test_train_model_rejects_empty_train_loader(tmp_path, adjacency):
    dataset = _dataset()
    train_loader = DataLoader(torch.utils.data.Subset(dataset, []), batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    with pytest.raises(ValueError, match="train_loader"):
        train_model(
            model=TinyKoopmanModel(),
            train_loader=train_loader,
            val_loader=val_loader,
            adjacency=adjacency,
            run_dir=tmp_path,
            max_epochs=1,
            learning_rate=0.01,
            patience=1,
            device="cpu",
            uses_koopman=True,
        )


def test_train_model_rejects_training_prediction_target_shape_mismatch(
    tmp_path, adjacency
):
    dataset = _dataset()
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)

    with pytest.raises(ValueError, match="shape"):
        train_model(
            model=MismatchedShapeModel(),
            train_loader=train_loader,
            val_loader=val_loader,
            adjacency=adjacency,
            run_dir=tmp_path,
            max_epochs=1,
            learning_rate=0.01,
            patience=1,
            device="cpu",
            uses_koopman=True,
        )


def test_train_model_rejects_validation_prediction_target_shape_mismatch(
    tmp_path, adjacency
):
    train_dataset = _dataset()
    val_dataset = _dataset()
    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    with pytest.raises(ValueError, match="shape"):
        train_model(
            model=MismatchedShapeModel(matching_calls=len(train_loader)),
            train_loader=train_loader,
            val_loader=val_loader,
            adjacency=adjacency,
            run_dir=tmp_path,
            max_epochs=1,
            learning_rate=0.01,
            patience=1,
            device="cpu",
            uses_koopman=False,
        )
