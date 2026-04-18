from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from koopman_gat_lstm.cli import formal as formal_cli
from koopman_gat_lstm.experiments.formal import run_formal_experiment


def test_run_formal_experiment_orchestrates_both_models_and_writes_summary(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    dataset_artifact_path = tmp_path / "dataset.npz"
    output_root = tmp_path / "formal-output"
    selected_case_id = "case-42"

    fake_config = SimpleNamespace(
        seed=42,
        export=SimpleNamespace(key_nodes=[1, 2], selected_case="case-default"),
        task=SimpleNamespace(input_start=5.0, sample_rate_hz=100),
        training=SimpleNamespace(
            batch_size=8,
            max_epochs=12,
            learning_rate=0.001,
            patience=3,
            device="cpu",
        ),
    )
    fake_artifact = SimpleNamespace(split={"train": ["case-a"], "val": ["case-b"], "test": ["case-c"]})

    call_log: list[tuple[str, SimpleNamespace]] = []

    def fake_load_config(path):
        assert path == config_path
        return fake_config

    def fake_load_dataset_artifact(path):
        assert path == dataset_artifact_path
        return fake_artifact

    def fake_run_full_training(args):
        call_log.append(("train", args))
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (args.run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (args.run_dir / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
        (args.run_dir / "metrics" / "train_result.json").write_text("{}", encoding="utf-8")
        return args.run_dir

    def _write_eval_artifacts(run_dir: Path) -> None:
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (run_dir / "cases" / selected_case_id).mkdir(parents=True, exist_ok=True)
        y_true = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        y_pred = y_true + 0.25
        np.savez(
            run_dir / "predictions" / "test_predictions.npz",
            y_pred=y_pred,
            y_true=y_true,
            case_ids=np.asarray([selected_case_id], dtype=str),
        )
        np.save(run_dir / "cases" / selected_case_id / "layer1_entropy.npy", np.array([0.5, 0.25]))
        np.save(run_dir / "cases" / selected_case_id / "layer2_entropy.npy", np.array([0.2, 0.1]))

    def fake_run_full_evaluation(args):
        call_log.append(("eval", args))
        _write_eval_artifacts(args.run_dir)
        if args.run_dir == output_root / "koopman":
            return {"selected_case_id": selected_case_id}
        assert args.selected_case == selected_case_id
        return {"selected_case_id": selected_case_id}

    monkeypatch.setattr("koopman_gat_lstm.experiments.formal.load_config", fake_load_config)
    monkeypatch.setattr("koopman_gat_lstm.experiments.formal.load_dataset_artifact", fake_load_dataset_artifact)
    monkeypatch.setattr("koopman_gat_lstm.experiments.formal.train_cli.run_full_training", fake_run_full_training)
    monkeypatch.setattr("koopman_gat_lstm.experiments.formal.eval_cli.run_full_evaluation", fake_run_full_evaluation)

    summary = run_formal_experiment(
        config_path=config_path,
        dataset_artifact=dataset_artifact_path,
        output_root=output_root,
        max_epochs=4,
        batch_size=16,
        selected_case="auto",
    )

    assert summary["selected_case_id"] == selected_case_id
    commands_path = output_root / "logs" / "formal_commands.json"
    summary_path = output_root / "comparison" / "formal_run_summary.json"
    commands = json.loads(commands_path.read_text(encoding="utf-8"))
    summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
    assert commands["selected_case_id"] == selected_case_id
    assert [command["name"] for command in commands["commands"]] == [
        "koopman_train",
        "standard_train",
        "koopman_eval",
        "standard_eval",
    ]
    assert commands["commands"][2]["args"]["selected_case"] == "auto"
    assert commands["commands"][3]["args"]["selected_case"] == selected_case_id
    assert summary_json["selected_case_id"] == selected_case_id
    assert summary_json["dataset_artifact"] == str(dataset_artifact_path.resolve())
    assert summary_json["run_directories"]["root"] == str(output_root.resolve())
    assert summary_json["training_config"] == {
        "batch_size": 16,
        "max_epochs": 4,
        "learning_rate": 0.001,
        "patience": 3,
        "device": "cpu",
        "seed": 42,
    }
    assert summary_json["comparison_paths"]["overall_metrics_comparison"].endswith(
        "overall_metrics_comparison.csv"
    )
    assert summary_json["smoke_note"] == "outputs/koopman_short is not formal output"
    standard_eval_calls = [
        args for kind, args in call_log if kind == "eval" and args.run_dir == output_root / "standard"
    ]
    assert len(standard_eval_calls) == 1
    assert standard_eval_calls[0].selected_case == selected_case_id
    assert (output_root / "comparison" / "overall_metrics_comparison.csv").is_file()


def test_run_formal_experiment_writes_commands_before_standard_case_mismatch(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    dataset_artifact_path = tmp_path / "dataset.npz"
    output_root = tmp_path / "formal-output"
    koopman_selected_case_id = "case-42"

    fake_config = SimpleNamespace(
        seed=42,
        export=SimpleNamespace(key_nodes=[1, 2], selected_case="case-default"),
        task=SimpleNamespace(input_start=5.0, sample_rate_hz=100),
        training=SimpleNamespace(
            batch_size=8,
            max_epochs=12,
            learning_rate=0.001,
            patience=3,
            device="cpu",
        ),
    )
    fake_artifact = SimpleNamespace(split={"train": ["case-a"], "val": ["case-b"], "test": ["case-c"]})

    def fake_load_config(path):
        assert path == config_path
        return fake_config

    def fake_load_dataset_artifact(path):
        assert path == dataset_artifact_path
        return fake_artifact

    def fake_run_full_training(args):
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (args.run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (args.run_dir / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
        (args.run_dir / "metrics" / "train_result.json").write_text("{}", encoding="utf-8")
        return args.run_dir

    def _write_eval_artifacts(run_dir: Path) -> None:
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (run_dir / "cases" / koopman_selected_case_id).mkdir(parents=True, exist_ok=True)
        y_true = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        y_pred = y_true + 0.25
        np.savez(
            run_dir / "predictions" / "test_predictions.npz",
            y_pred=y_pred,
            y_true=y_true,
            case_ids=np.asarray([koopman_selected_case_id], dtype=str),
        )
        np.save(run_dir / "cases" / koopman_selected_case_id / "layer1_entropy.npy", np.array([0.5, 0.25]))
        np.save(run_dir / "cases" / koopman_selected_case_id / "layer2_entropy.npy", np.array([0.2, 0.1]))

    def fake_run_full_evaluation(args):
        _write_eval_artifacts(args.run_dir)
        if args.run_dir == output_root / "koopman":
            return {"selected_case_id": koopman_selected_case_id}
        return {"selected_case_id": "case-other"}

    monkeypatch.setattr("koopman_gat_lstm.experiments.formal.load_config", fake_load_config)
    monkeypatch.setattr("koopman_gat_lstm.experiments.formal.load_dataset_artifact", fake_load_dataset_artifact)
    monkeypatch.setattr("koopman_gat_lstm.experiments.formal.train_cli.run_full_training", fake_run_full_training)
    monkeypatch.setattr("koopman_gat_lstm.experiments.formal.eval_cli.run_full_evaluation", fake_run_full_evaluation)

    with pytest.raises(ValueError, match="different selected case"):
        run_formal_experiment(
            config_path=config_path,
            dataset_artifact=dataset_artifact_path,
            output_root=output_root,
            max_epochs=4,
            batch_size=16,
            selected_case="auto",
        )

    commands_path = output_root.resolve() / "logs" / "formal_commands.json"
    assert commands_path.is_file()
    commands = json.loads(commands_path.read_text(encoding="utf-8"))
    assert commands["selected_case_id"] == koopman_selected_case_id
    assert commands["commands"][3]["args"]["selected_case"] == koopman_selected_case_id


def test_formal_cli_forwards_arguments(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    dataset_artifact_path = tmp_path / "dataset.npz"
    monkeypatch.chdir(tmp_path)
    output_root = Path("nested") / ".." / "formal-output"
    expected_output_root = (tmp_path / "formal-output").resolve()
    captured_kwargs = {}
    call_count = {"value": 0}

    def fake_run_formal_experiment(**kwargs):
        call_count["value"] += 1
        captured_kwargs.update(kwargs)
        summary_path = output_root / "comparison" / "formal_run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("{}", encoding="utf-8")
        return {"selected_case_id": "case-a"}

    monkeypatch.setattr(formal_cli, "run_formal_experiment", fake_run_formal_experiment)

    return_code = formal_cli.main(
        [
            "--config",
            str(config_path),
            "--dataset-artifact",
            str(dataset_artifact_path),
            "--output-root",
            str(output_root),
            "--max-epochs",
            "7",
            "--batch-size",
            "3",
            "--selected-case",
            "case-a",
        ]
    )

    assert return_code == 0
    assert call_count["value"] == 1
    assert captured_kwargs == {
        "config_path": config_path,
        "dataset_artifact": dataset_artifact_path,
        "output_root": expected_output_root,
        "max_epochs": 7,
        "batch_size": 3,
        "selected_case": "case-a",
    }
    out = capsys.readouterr().out.splitlines()
    assert out == [
        str(expected_output_root / "comparison" / "formal_run_summary.json"),
        "case-a",
    ]


@pytest.mark.parametrize("invalid_value", ["0", "abc"])
def test_formal_cli_rejects_invalid_positive_ints(invalid_value):
    parser = formal_cli.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "--config",
                "config.yaml",
                "--max-epochs",
                invalid_value,
            ]
        )

    assert excinfo.value.code == 2
