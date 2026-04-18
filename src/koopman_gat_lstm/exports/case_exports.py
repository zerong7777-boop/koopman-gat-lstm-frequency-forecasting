from __future__ import annotations

from pathlib import Path


def build_case_dir(run_dir: Path | str, case_id: str) -> Path:
    """Create and return the case-specific export directory for a run."""
    if not case_id or case_id in {".", ".."}:
        raise ValueError("case_id must be a non-empty leaf directory name")
    if any(token in case_id for token in ("/", "\\", ":")):
        raise ValueError("case_id must not contain path separators or drive qualifiers")

    cases_root = Path(run_dir) / "cases"
    case_dir = cases_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir
