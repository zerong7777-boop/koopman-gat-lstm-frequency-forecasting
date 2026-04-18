import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
from koopman_gat_lstm.cli import eval as eval_cli
from koopman_gat_lstm.cli import train as train_cli


def test_train_and_eval_smoke_create_run_case_and_entropy_artifacts():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "default.yaml"
    run_dir = Path(__file__).resolve().parent / "artifacts" / f"task8-smoke-run-{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)

    assert train_cli.main(["--config", str(config_path), "--smoke", "--run-dir", str(run_dir)]) == 0

    train_case_dir = run_dir / "cases" / "smoke-train"
    train_entropy = train_case_dir / "entropy_curve.npy"
    assert run_dir.is_dir()
    assert train_case_dir.is_dir()
    assert train_entropy.is_file()
    train_curve = np.load(train_entropy)
    assert train_curve.shape == (1,)
    assert np.isfinite(train_curve).all()

    assert eval_cli.main(["--run-dir", str(run_dir), "--smoke"]) == 0

    eval_case_dir = run_dir / "cases" / "smoke-eval"
    eval_entropy = eval_case_dir / "entropy_curve.npy"
    assert eval_case_dir.is_dir()
    assert eval_entropy.is_file()
    eval_curve = np.load(eval_entropy)
    assert eval_curve.shape == (1,)
    assert np.isfinite(eval_curve).all()


def test_train_cli_module_help_exposes_smoke_flag():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "koopman_gat_lstm.cli.train", "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--smoke" in result.stdout


def test_cli_non_smoke_exits_without_traceback():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    commands = [
        [sys.executable, "-m", "koopman_gat_lstm.cli.train", "--config", "configs/default.yaml"],
        [sys.executable, "-m", "koopman_gat_lstm.cli.eval", "--run-dir", "."],
    ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert "Traceback" not in result.stderr
        assert "--smoke is currently required" in result.stderr
