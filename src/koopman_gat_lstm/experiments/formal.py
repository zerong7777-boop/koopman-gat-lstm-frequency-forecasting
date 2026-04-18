from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from koopman_gat_lstm.cli import eval as eval_cli
from koopman_gat_lstm.cli import train as train_cli
from koopman_gat_lstm.config import load_config
from koopman_gat_lstm.data.artifacts import load_dataset_artifact
from koopman_gat_lstm.eval.comparison import write_formal_comparison_artifacts


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return {key: _serialize_value(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return value


def _namespace_payload(namespace: SimpleNamespace) -> dict[str, Any]:
    return {key: _serialize_value(value) for key, value in vars(namespace).items()}


def _git_short_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    short_head = result.stdout.strip()
    return short_head or None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_formal_experiment(
    *,
    config_path: Path,
    dataset_artifact: Path,
    output_root: Path,
    max_epochs: int,
    batch_size: int | None = None,
    selected_case: str = "auto",
) -> dict[str, Any]:
    """Run both formal model experiments and return the formal run summary."""

    config_path = Path(config_path).resolve()
    dataset_artifact = Path(dataset_artifact).resolve()
    output_root = Path(output_root).resolve()
    koopman_run_dir = output_root / "koopman"
    standard_run_dir = output_root / "standard"
    comparison_dir = output_root / "comparison"
    logs_dir = output_root / "logs"
    for path in (koopman_run_dir, standard_run_dir, comparison_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    artifact = load_dataset_artifact(dataset_artifact)
    effective_batch_size = batch_size or config.training.batch_size

    koopman_train_args = SimpleNamespace(
        config=config_path,
        dataset_artifact=dataset_artifact,
        run_dir=koopman_run_dir,
        model_type="koopman",
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    train_cli.run_full_training(koopman_train_args)

    standard_train_args = SimpleNamespace(
        config=config_path,
        dataset_artifact=dataset_artifact,
        run_dir=standard_run_dir,
        model_type="standard",
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    train_cli.run_full_training(standard_train_args)

    koopman_eval_args = SimpleNamespace(
        config=config_path,
        dataset_artifact=dataset_artifact,
        run_dir=koopman_run_dir,
        model_type="koopman",
        checkpoint=koopman_run_dir / "checkpoints" / "best.pt",
        selected_case=selected_case,
    )
    koopman_eval_result = eval_cli.run_full_evaluation(koopman_eval_args)
    selected_case_id = str(koopman_eval_result["selected_case_id"])

    standard_eval_args = SimpleNamespace(
        config=config_path,
        dataset_artifact=dataset_artifact,
        run_dir=standard_run_dir,
        model_type="standard",
        checkpoint=standard_run_dir / "checkpoints" / "best.pt",
        selected_case=selected_case_id,
    )
    standard_eval_result = eval_cli.run_full_evaluation(standard_eval_args)
    standard_selected_case_id = str(standard_eval_result["selected_case_id"])
    commands_payload = {
        "selected_case_id": selected_case_id,
        "commands": [
            {
                "name": "koopman_train",
                "function": "train_cli.run_full_training",
                "args": _namespace_payload(koopman_train_args),
            },
            {
                "name": "standard_train",
                "function": "train_cli.run_full_training",
                "args": _namespace_payload(standard_train_args),
            },
            {
                "name": "koopman_eval",
                "function": "eval_cli.run_full_evaluation",
                "args": _namespace_payload(koopman_eval_args),
            },
            {
                "name": "standard_eval",
                "function": "eval_cli.run_full_evaluation",
                "args": _namespace_payload(standard_eval_args),
            },
        ],
    }
    _write_json(logs_dir / "formal_commands.json", commands_payload)

    if standard_selected_case_id != selected_case_id:
        raise ValueError(
            "standard evaluation returned a different selected case: "
            f"expected {selected_case_id}, got {standard_selected_case_id}"
        )

    comparison_result = write_formal_comparison_artifacts(
        koopman_run_dir=koopman_run_dir,
        standard_run_dir=standard_run_dir,
        comparison_dir=comparison_dir,
        selected_case_id=selected_case_id,
        key_nodes=config.export.key_nodes,
        input_start_s=float(config.task.input_start),
        sample_rate_hz=int(config.task.sample_rate_hz),
    )

    timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary = {
        "timestamp_utc": timestamp_utc,
        "git_short_head": _git_short_head(Path(__file__).resolve().parents[3]),
        "dataset_artifact": str(dataset_artifact),
        "split_sizes": {split_name: len(case_ids) for split_name, case_ids in artifact.split.items()},
        "training_config": {
            "batch_size": effective_batch_size,
            "max_epochs": max_epochs,
            "learning_rate": config.training.learning_rate,
            "patience": config.training.patience,
            "device": config.training.device,
            "seed": config.seed,
        },
        "run_directories": {
            "root": str(output_root),
            "koopman": str(koopman_run_dir),
            "standard": str(standard_run_dir),
            "comparison": str(comparison_dir),
            "logs": str(logs_dir),
        },
        "selected_case_id": selected_case_id,
        "comparison_paths": comparison_result["paths"],
        "smoke_note": "outputs/koopman_short is not formal output",
    }
    _write_json(comparison_dir / "formal_run_summary.json", summary)
    return summary
