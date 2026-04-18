from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
import math
from typing import Any

import yaml


def steps_from_interval(start: float, end: float, sample_rate_hz: int) -> int:
    """Convert a left-closed, right-open time interval into a sample count."""
    return int(round((end - start) * sample_rate_hz))


@dataclass
class TaskConfig:
    input_start: float
    input_end: float
    forecast_start: float
    forecast_end: float
    sample_rate_hz: int
    node_order: list[str]
    input_steps: int = field(init=False)
    forecast_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.input_steps = steps_from_interval(self.input_start, self.input_end, self.sample_rate_hz)
        self.forecast_steps = steps_from_interval(self.forecast_start, self.forecast_end, self.sample_rate_hz)


@dataclass
class PathConfig:
    frequency_workbook: Path
    koopman_workbook: Path
    adjacency_workbook: Path
    output_dir: Path


@dataclass
class ModelConfig:
    gat_hidden_dim: int
    gat_heads_layer1: int
    gat_heads_layer2: int
    lstm_hidden_dim: int


@dataclass
class TrainingConfig:
    batch_size: int
    max_epochs: int
    learning_rate: float
    patience: int
    device: str


@dataclass
class ExportConfig:
    key_nodes: list[int]
    selected_case: str | None = None


@dataclass
class AppConfig:
    seed: int
    split: dict
    task: TaskConfig
    paths: PathConfig
    model: ModelConfig
    training: TrainingConfig
    export: ExportConfig


def _require_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a mapping")
    return value


def _parse_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive int")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive int") from exc
    if numeric <= 0 or not numeric.is_integer():
        raise ValueError(f"{field_name} must be a positive int")
    return int(numeric)


def _parse_strict_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive int")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive int")
    return value


def _parse_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive float")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive float") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{field_name} must be a positive float")
    return numeric


def _parse_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _parse_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be None or a string")
    return value


def _parse_key_nodes(value: Any) -> list[int]:
    if not isinstance(value, list):
        raise ValueError("export.key_nodes must be a list of positive ints")
    return [_parse_strict_positive_int(item, "export.key_nodes item") for item in value]


def _load_paths_config(data: Mapping[str, Any]) -> PathConfig:
    return PathConfig(
        frequency_workbook=Path(data["frequency_workbook"]),
        koopman_workbook=Path(data["koopman_workbook"]),
        adjacency_workbook=Path(data["adjacency_workbook"]),
        output_dir=Path(data["output_dir"]),
    )


def _load_model_config(data: Mapping[str, Any]) -> ModelConfig:
    return ModelConfig(
        gat_hidden_dim=_parse_positive_int(data["gat_hidden_dim"], "model.gat_hidden_dim"),
        gat_heads_layer1=_parse_positive_int(data["gat_heads_layer1"], "model.gat_heads_layer1"),
        gat_heads_layer2=_parse_positive_int(data["gat_heads_layer2"], "model.gat_heads_layer2"),
        lstm_hidden_dim=_parse_positive_int(data["lstm_hidden_dim"], "model.lstm_hidden_dim"),
    )


def _load_training_config(data: Mapping[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        batch_size=_parse_positive_int(data["batch_size"], "training.batch_size"),
        max_epochs=_parse_positive_int(data["max_epochs"], "training.max_epochs"),
        learning_rate=_parse_positive_float(data["learning_rate"], "training.learning_rate"),
        patience=_parse_positive_int(data["patience"], "training.patience"),
        device=_parse_non_empty_str(data["device"], "training.device"),
    )


def _load_export_config(data: Mapping[str, Any]) -> ExportConfig:
    return ExportConfig(
        key_nodes=_parse_key_nodes(data["key_nodes"]),
        selected_case=_parse_optional_str(data.get("selected_case"), "export.selected_case"),
    )


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("config root must be a mapping")
    task_data = _require_mapping(data["task"], "task")
    task = TaskConfig(
        input_start=task_data["input_start"],
        input_end=task_data["input_end"],
        forecast_start=task_data["forecast_start"],
        forecast_end=task_data["forecast_end"],
        sample_rate_hz=_parse_strict_positive_int(task_data["sample_rate_hz"], "task.sample_rate_hz"),
        node_order=task_data["node_order"],
    )
    paths = _load_paths_config(_require_mapping(data["paths"], "paths"))
    model = _load_model_config(_require_mapping(data["model"], "model"))
    training = _load_training_config(_require_mapping(data["training"], "training"))
    export = _load_export_config(_require_mapping(data["export"], "export"))
    return AppConfig(
        seed=data["seed"],
        split=data["split"],
        task=task,
        paths=paths,
        model=model,
        training=training,
        export=export,
    )
