from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from koopman_gat_lstm.config import load_config
from koopman_gat_lstm.exports.case_exports import build_case_dir
from koopman_gat_lstm.exports.entropy_exports import compute_layer_entropy_curve

_MISSING_FULL_ARGS_MESSAGE = (
    "--smoke is currently required for this minimal invocation; full training "
    "requires an existing dataset artifact"
)


def _smoke_attention() -> np.ndarray:
    return np.array(
        [
            [
                [[0.5, 0.5], [0.25, 0.75]],
            ]
        ],
        dtype=np.float64,
    )


def _write_smoke_entropy(case_dir: Path) -> Path:
    entropy_curve = compute_layer_entropy_curve(_smoke_attention())
    artifact_path = case_dir / "entropy_curve.npy"
    np.save(artifact_path, entropy_curve)
    return artifact_path


def run_smoke(config_path: Path, run_dir: Path | None = None) -> Path:
    config = load_config(config_path)
    if run_dir is None:
        run_dir = Path(tempfile.mkdtemp(prefix="koopman-gat-lstm-smoke-"))

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "smoke_metrics.json").write_text(
        json.dumps({"seed": config.seed, "node_count": len(config.task.node_order)}, indent=2),
        encoding="utf-8",
    )

    case_dir = build_case_dir(run_dir, "smoke-train")
    _write_smoke_entropy(case_dir)
    return run_dir


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _dataset_artifact_path(config, config_path: Path, requested_path: Path | None) -> Path:
    if requested_path is not None:
        return requested_path
    output_dir = config.paths.output_dir
    if not output_dir.is_absolute():
        output_dir = config_path.resolve().parent / output_dir
    return output_dir / "dataset.npz"


def _build_model(model_type: str, config):
    if model_type == "koopman":
        from koopman_gat_lstm.models.koopman import KoopmanGATLSTM as Model
    else:
        from koopman_gat_lstm.models.standard import StandardGATLSTM as Model

    return Model(
        num_nodes=len(config.task.node_order),
        input_feature_dim=1,
        gat_hidden_dim=config.model.gat_hidden_dim,
        gat_heads_layer1=config.model.gat_heads_layer1,
        gat_heads_layer2=config.model.gat_heads_layer2,
        lstm_hidden_dim=config.model.lstm_hidden_dim,
        forecast_steps=config.task.forecast_steps,
    )


def run_full_training(args: argparse.Namespace) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from koopman_gat_lstm.data.artifacts import load_dataset_artifact
    from koopman_gat_lstm.data.torch_dataset import ForecastDataset
    from koopman_gat_lstm.train.trainer import train_model

    config = load_config(args.config)
    artifact_path = _dataset_artifact_path(config, args.config, args.dataset_artifact)
    artifact = load_dataset_artifact(artifact_path)

    torch.manual_seed(config.seed)
    model = _build_model(args.model_type, config)
    batch_size = args.batch_size or config.training.batch_size
    train_dataset = ForecastDataset(artifact, artifact.split["train"])
    val_dataset = ForecastDataset(artifact, artifact.split["val"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    adjacency = torch.as_tensor(artifact.adjacency, dtype=torch.float32)

    run_dir = args.run_dir or config.paths.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        adjacency=adjacency,
        run_dir=run_dir,
        max_epochs=args.max_epochs or config.training.max_epochs,
        learning_rate=config.training.learning_rate,
        patience=config.training.patience,
        device=_resolve_device(config.training.device),
        uses_koopman=args.model_type == "koopman",
    )

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "train_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Koopman GAT-LSTM experiments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment config YAML.")
    parser.add_argument("--smoke", action="store_true", help="Run a lightweight CLI smoke check.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run directory for smoke output.")
    parser.add_argument(
        "--model-type",
        choices=("standard", "koopman"),
        default="koopman",
        help="Model family to train.",
    )
    parser.add_argument(
        "--dataset-artifact",
        type=Path,
        default=None,
        help="Path to the materialized dataset artifact; defaults to paths.output_dir/dataset.npz.",
    )
    parser.add_argument(
        "--max-epochs",
        type=_positive_int,
        default=None,
        help="Override the training.max_epochs config value.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="Override the training.batch_size config value.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        run_dir = run_smoke(args.config, args.run_dir)
        print(run_dir)
        return 0

    config = load_config(args.config)
    artifact_path = _dataset_artifact_path(config, args.config, args.dataset_artifact)
    if not artifact_path.is_file():
        parser.error(f"{_MISSING_FULL_ARGS_MESSAGE}: {artifact_path}")

    run_dir = run_full_training(args)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
