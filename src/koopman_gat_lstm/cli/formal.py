from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from koopman_gat_lstm.experiments.formal import run_formal_experiment


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formal Koopman vs Standard GAT-LSTM experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-artifact", type=Path, default=Path("outputs/dataset/dataset.npz"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/formal"))
    parser.add_argument("--max-epochs", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--selected-case", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    summary = run_formal_experiment(
        config_path=args.config,
        dataset_artifact=args.dataset_artifact,
        output_root=output_root,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        selected_case=args.selected_case,
    )
    print(output_root / "comparison" / "formal_run_summary.json")
    print(summary["selected_case_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
