from pathlib import Path
from unittest.mock import patch

import pytest

from koopman_gat_lstm.config import load_config


def test_default_config_exposes_task_contract():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

    cfg = load_config(config_path)

    assert cfg.seed == 42
    assert cfg.task.input_steps == 40
    assert cfg.task.forecast_steps == 260
    assert "input_steps" in cfg.task.__dict__
    assert "forecast_steps" in cfg.task.__dict__
    assert cfg.task.node_order[0] == "BUS1"
    assert cfg.task.node_order[-1] == "BUS39"


def test_default_config_exposes_runtime_paths_and_model_defaults():
    cfg = load_config("configs/default.yaml")

    assert cfg.paths.frequency_workbook.name == "frequency_workbook.xlsx"
    assert cfg.paths.koopman_workbook.name == "koopman_energy.xlsx"
    assert cfg.paths.adjacency_workbook.name == "adjacency_matrix.xlsx"
    assert cfg.paths.output_dir.name == "outputs"
    assert cfg.model.gat_hidden_dim > 0
    assert cfg.model.gat_heads_layer1 > 0
    assert cfg.model.gat_heads_layer2 > 0
    assert cfg.model.lstm_hidden_dim == 64
    assert cfg.training.max_epochs >= 1
    assert cfg.training.batch_size >= 1
    assert cfg.training.learning_rate == 0.001
    assert cfg.training.patience == 10
    assert cfg.training.device == "auto"
    assert cfg.export.key_nodes == [1, 9, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    assert cfg.export.selected_case is None
    assert cfg.task.sample_rate_hz == 100


def test_config_accepts_string_learning_rate_and_coerces_to_float():
    yaml_text = """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: 100
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: "1e-3"
  patience: 10
  device: auto
export:
  key_nodes: [1, 9, 31, 32, 33, 34, 35, 36, 37, 38, 39]
"""

    with patch.object(Path, "read_text", return_value=yaml_text):
        cfg = load_config("configs/default.yaml")

    assert cfg.training.learning_rate == pytest.approx(0.001)
    assert isinstance(cfg.training.learning_rate, float)


@pytest.mark.parametrize(
    "yaml_text, expected_message",
    [
        (
            """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: 100
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: nan
  patience: 10
  device: auto
export:
  key_nodes: [1, 2, 3]
""",
            "training.learning_rate",
        ),
        (
            """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: 100
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: inf
  patience: 10
  device: auto
export:
  key_nodes: [1, 2, 3]
""",
            "training.learning_rate",
        ),
        (
            """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: 100
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: 0.001
  patience: 10
  device: auto
export:
  key_nodes: [1, nope, 3]
""",
            "export.key_nodes",
        ),
        (
            """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: 100
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 0
  max_epochs: 50
  learning_rate: 0.001
  patience: 10
  device: auto
export:
  key_nodes: [1, 2, 3]
""",
            "training.batch_size",
        ),
        (
            """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: 100
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: 0.001
  patience: 10
  device: auto
export:
  key_nodes: ["1", 2]
""",
            "export.key_nodes",
        ),
        (
            """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: 100
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: 0.001
  patience: 10
  device: auto
export:
  key_nodes: [1.0, 2]
""",
            "export.key_nodes",
        ),
        (
            """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: 0
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: 0.001
  patience: 10
  device: auto
export:
  key_nodes: [1, 2, 3]
""",
            "task.sample_rate_hz",
        ),
        (
            """
seed: 42
split:
  train: 0.7
  val: 0.1
  test: 0.2
task:
  input_start: 5.0
  input_end: 5.4
  forecast_start: 5.4
  forecast_end: 8.0
  sample_rate_hz: -100
  node_order: [BUS1, BUS2]
paths:
  frequency_workbook: data/raw/frequency_workbook.xlsx
  koopman_workbook: data/raw/koopman_energy.xlsx
  adjacency_workbook: data/raw/adjacency_matrix.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 16
  gat_heads_layer1: 4
  gat_heads_layer2: 4
  lstm_hidden_dim: 64
training:
  batch_size: 16
  max_epochs: 50
  learning_rate: 0.001
  patience: 10
  device: auto
export:
  key_nodes: [1, 2, 3]
""",
            "task.sample_rate_hz",
        ),
    ],
)
def test_config_rejects_invalid_values(yaml_text, expected_message):
    with patch.object(Path, "read_text", return_value=yaml_text):
        with pytest.raises(ValueError, match=expected_message):
            load_config("configs/default.yaml")
