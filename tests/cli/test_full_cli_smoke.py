import os
from pathlib import Path
import subprocess
import sys
import shutil
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from unittest.mock import MagicMock, patch

from koopman_gat_lstm.data.artifacts import MaterializedDataset, save_dataset_artifact
from koopman_gat_lstm.cli import eval as eval_cli


def _subprocess_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return env


def test_full_cli_trains_and_evaluates_with_config_relative_dataset_artifact():
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_root = Path(__file__).resolve().parent / "artifacts" / f"task8-full-cli-smoke-{uuid4().hex}"
    config_dir = artifacts_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir = config_dir / "outputs"
    run_dir = artifacts_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact = MaterializedDataset(
        case_ids=np.array(["case-train-a", "case-train-b", "case-val", "case-test"]),
        x=np.arange(16, dtype=np.float32).reshape(4, 2, 2) / 10.0,
        y=np.arange(16, 32, dtype=np.float32).reshape(4, 2, 2) / 10.0,
        koopman=np.ones((4, 2), dtype=np.float32),
        adjacency=np.eye(2, dtype=np.float32),
        frequency_mean=np.zeros(2, dtype=np.float32),
        frequency_std=np.ones(2, dtype=np.float32),
        split={
            "train": ["case-train-a", "case-train-b"],
            "val": ["case-val"],
            "test": ["case-test"],
        },
    )
    save_dataset_artifact(output_dir, artifact)

    config_path = config_dir / "tiny.yaml"
    config_path.write_text(
        """
paths:
  frequency_workbook: unused.xlsx
  koopman_workbook: unused.xlsx
  adjacency_workbook: unused.xlsx
  output_dir: outputs
model:
  gat_hidden_dim: 2
  gat_heads_layer1: 1
  gat_heads_layer2: 1
  lstm_hidden_dim: 3
training:
  batch_size: 2
  max_epochs: 1
  learning_rate: 0.001
  patience: 1
  device: cpu
export:
  key_nodes: [1]
  selected_case: case-test
seed: 42
split:
  train: 0.5
  val: 0.25
  test: 0.25
task:
  input_start: 0.0
  input_end: 0.2
  forecast_start: 0.2
  forecast_end: 0.4
  sample_rate_hz: 10
  node_order: [BUS1, BUS2]
""".lstrip(),
        encoding="utf-8",
    )

    try:
        train_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "koopman_gat_lstm.cli.train",
                "--config",
                str(config_path),
                "--model-type",
                "standard",
                "--max-epochs",
                "1",
                "--run-dir",
                str(run_dir),
            ],
            cwd=repo_root,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )

        checkpoint = run_dir / "checkpoints" / "best.pt"
        assert train_result.returncode == 0, train_result.stderr
        assert checkpoint.is_file()

        eval_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "koopman_gat_lstm.cli.eval",
                "--run-dir",
                str(run_dir),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--model-type",
                "standard",
                "--selected-case",
                "auto",
            ],
            cwd=repo_root,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )

        assert eval_result.returncode == 0, eval_result.stderr
        assert (run_dir / "metrics" / "test_metrics.json").is_file()
        assert (run_dir / "predictions" / "test_predictions.npz").is_file()
        case_dir = run_dir / "cases" / "case-test"
        assert list(case_dir.glob("*_key_node_comparison.xlsx"))
        assert (case_dir / "node_1_prediction.png").is_file()
    finally:
        shutil.rmtree(artifacts_root, ignore_errors=True)


def test_train_cli_exposes_full_experiment_arguments():
    result = subprocess.run(
        [sys.executable, "-m", "koopman_gat_lstm.cli.train", "--help"],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--model-type" in result.stdout
    assert "--dataset-artifact" in result.stdout
    assert "--max-epochs" in result.stdout


def test_eval_cli_exposes_full_experiment_arguments():
    result = subprocess.run(
        [sys.executable, "-m", "koopman_gat_lstm.cli.eval", "--help"],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--selected-case" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--dataset-artifact" in result.stdout


def test_eval_run_full_evaluation_treats_auto_selected_case_as_default():
    fake_config = SimpleNamespace(
        paths=SimpleNamespace(output_dir=Path("outputs")),
        training=SimpleNamespace(device="cpu", batch_size=2),
        model=SimpleNamespace(
            gat_hidden_dim=2,
            gat_heads_layer1=1,
            gat_heads_layer2=1,
            lstm_hidden_dim=3,
        ),
        task=SimpleNamespace(node_order=["BUS1", "BUS2"], forecast_steps=1),
        export=SimpleNamespace(key_nodes=[1], selected_case="case-test"),
    )
    fake_artifact = SimpleNamespace(
        split={"test": ["case-test"]},
        adjacency=np.eye(2, dtype=np.float32),
    )
    fake_model = MagicMock()
    fake_model.load_state_dict.return_value = None
    evaluate_model = MagicMock(return_value={"selected_case_id": "case-test"})
    export_selected_case_key_nodes = MagicMock()

    args = SimpleNamespace(
        config=Path("config.yaml"),
        checkpoint=Path("checkpoint.pt"),
        dataset_artifact=None,
        model_type="standard",
        run_dir=Path("run"),
        selected_case="auto",
    )

    with patch.object(eval_cli, "load_config", return_value=fake_config), patch.object(
        eval_cli, "_build_model", return_value=fake_model
    ), patch(
        "koopman_gat_lstm.data.artifacts.load_dataset_artifact", return_value=fake_artifact
    ), patch(
        "koopman_gat_lstm.data.torch_dataset.ForecastDataset", return_value="dataset"
    ), patch("koopman_gat_lstm.eval.evaluator.evaluate_model", evaluate_model), patch(
        "torch.load", return_value={"model_state_dict": {}}
    ), patch("torch.utils.data.DataLoader", return_value="loader"), patch.object(
        eval_cli,
        "_export_selected_case_key_nodes",
        export_selected_case_key_nodes,
    ):
        result = eval_cli.run_full_evaluation(args)

    assert result["selected_case_id"] == "case-test"
    assert evaluate_model.call_count == 1
    assert evaluate_model.call_args.kwargs["selected_case_id"] is None
    export_selected_case_key_nodes.assert_called_once_with(
        Path("run"),
        result,
        fake_artifact,
        [1],
    )
