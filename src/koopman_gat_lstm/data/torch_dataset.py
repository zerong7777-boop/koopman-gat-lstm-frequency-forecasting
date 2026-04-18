from __future__ import annotations

import torch
from torch.utils.data import Dataset

from koopman_gat_lstm.data.artifacts import MaterializedDataset


class ForecastDataset(Dataset):
    """Torch dataset view over materialized forecast artifacts."""

    def __init__(self, artifact: MaterializedDataset, case_ids) -> None:
        self.artifact = artifact
        case_index = {str(case_id): idx for idx, case_id in enumerate(artifact.case_ids)}
        self.case_ids = [str(case_id) for case_id in case_ids]
        self.row_indices = [case_index[case_id] for case_id in self.case_ids]

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        row_index = self.row_indices[index]
        return {
            "case_id": self.case_ids[index],
            "x": torch.as_tensor(self.artifact.x[row_index], dtype=torch.float32),
            "y": torch.as_tensor(self.artifact.y[row_index], dtype=torch.float32),
            "koopman": torch.as_tensor(self.artifact.koopman[row_index], dtype=torch.float32),
        }
