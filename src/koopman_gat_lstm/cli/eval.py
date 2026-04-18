from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from koopman_gat_lstm.config import load_config
from koopman_gat_lstm.exports.case_exports import build_case_dir
from koopman_gat_lstm.exports.entropy_exports import compute_layer_entropy_curve
from koopman_gat_lstm.exports.key_nodes import export_key_node_comparisons

_MISSING_FULL_ARGS_MESSAGE = (
    "--smoke is currently required for this minimal invocation; full evaluation "
    "requires --config, --checkpoint, and an existing dataset artifact"
)


def _smoke_attention() -> np.ndarray:
    return np.array(
        [
            [
                [[0.5, 0.5], [0.5, 0.5]],
            ]
        ],
        dtype=np.float64,
    )


def run_smoke(run_dir: Path) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    case_dir = build_case_dir(run_dir, "smoke-eval")
    entropy_curve = compute_layer_entropy_curve(_smoke_attention())
    np.save(case_dir / "entropy_curve.npy", entropy_curve)
    return case_dir


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


def _export_selected_case_key_nodes(
    run_dir: Path,
    result: dict,
    artifact,
    key_nodes: Sequence[int],
) -> None:
    selected_case_id = str(result["selected_case_id"])
    predictions_path = run_dir / "predictions" / "test_predictions.npz"
    with np.load(predictions_path, allow_pickle=False) as predictions:
        case_ids = [str(case_id) for case_id in predictions["case_ids"]]
        try:
            selected_index = case_ids.index(selected_case_id)
        except ValueError as exc:
            raise ValueError(f"selected case id was not found in predictions: {selected_case_id}") from exc

        y_true_case = predictions["y_true"][selected_index]
        y_pred_case = predictions["y_pred"][selected_index]

    frequency_mean = np.asarray(artifact.frequency_mean)
    frequency_std = np.asarray(artifact.frequency_std)
    y_true_case = y_true_case * frequency_std + frequency_mean
    y_pred_case = y_pred_case * frequency_std + frequency_mean

    case_dir = build_case_dir(run_dir, selected_case_id)
    export_key_node_comparisons(case_dir, selected_case_id, y_true_case, y_pred_case, key_nodes)


def run_full_evaluation(args: argparse.Namespace) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from koopman_gat_lstm.data.artifacts import load_dataset_artifact
    from koopman_gat_lstm.data.torch_dataset import ForecastDataset
    from koopman_gat_lstm.eval.evaluator import evaluate_model

    config = load_config(args.config)
    artifact = load_dataset_artifact(_dataset_artifact_path(config, args.config, args.dataset_artifact))
    model = _build_model(args.model_type, config)
    device = _resolve_device(config.training.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_case_ids = artifact.split["test"]
    loader = DataLoader(
        ForecastDataset(artifact, test_case_ids),
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    if args.selected_case == "auto":
        selected_case = None
    elif args.selected_case is not None:
        selected_case = args.selected_case
    else:
        selected_case = config.export.selected_case
    result = evaluate_model(
        model=model,
        loader=loader,
        adjacency=artifact.adjacency,
        run_dir=args.run_dir,
        case_ids=test_case_ids,
        uses_koopman=args.model_type == "koopman",
        device=device,
        selected_case_id=selected_case,
    )
    _export_selected_case_key_nodes(args.run_dir, result, artifact, config.export.key_nodes)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Koopman GAT-LSTM experiment outputs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory to evaluate.")
    parser.add_argument("--smoke", action="store_true", help="Run a lightweight CLI smoke check.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the experiment config YAML.")
    parser.add_argument(
        "--model-type",
        choices=("standard", "koopman"),
        default="koopman",
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--dataset-artifact",
        type=Path,
        default=None,
        help="Path to the materialized dataset artifact; defaults to paths.output_dir/dataset.npz.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to the model checkpoint.")
    parser.add_argument(
        "--selected-case",
        default=None,
        help="Case ID to use for attention entropy exports; defaults to export.selected_case.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        case_dir = run_smoke(args.run_dir)
        print(case_dir)
        return 0

    if args.config is None:
        parser.error(_MISSING_FULL_ARGS_MESSAGE)
    if args.checkpoint is None or not args.checkpoint.is_file():
        parser.error(f"{_MISSING_FULL_ARGS_MESSAGE}: {args.checkpoint}")

    config = load_config(args.config)
    artifact_path = _dataset_artifact_path(config, args.config, args.dataset_artifact)
    if not artifact_path.is_file():
        parser.error(f"{_MISSING_FULL_ARGS_MESSAGE}: {artifact_path}")

    result = run_full_evaluation(args)
    print(result["selected_case_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
