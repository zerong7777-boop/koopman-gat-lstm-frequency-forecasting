import numpy as np
import pytest

from koopman_gat_lstm.exports.key_nodes import export_key_node_comparisons


def test_export_key_node_comparisons_writes_excel_and_plots(tmp_path):
    y_true = np.zeros((260, 39))
    y_pred = np.ones((260, 39))

    outputs = export_key_node_comparisons(tmp_path, "case-a", y_true, y_pred, [1, 9, 31])

    assert outputs["excel"].is_file()
    assert (tmp_path / "node_1_prediction.png").is_file()
    assert (tmp_path / "node_9_prediction.png").is_file()
    assert (tmp_path / "node_31_prediction.png").is_file()


@pytest.mark.parametrize("key_nodes", ([0, 1], [-1, 1], [40]))
def test_export_key_node_comparisons_rejects_invalid_key_nodes(tmp_path, key_nodes):
    y_true = np.zeros((260, 39))
    y_pred = np.ones((260, 39))

    with pytest.raises(ValueError, match="positive 1-based BUS numbers|exceeds available node columns"):
        export_key_node_comparisons(tmp_path, "case-a", y_true, y_pred, key_nodes)
