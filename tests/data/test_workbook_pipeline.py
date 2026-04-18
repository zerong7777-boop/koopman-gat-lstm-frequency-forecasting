import numpy as np
import pandas as pd

from koopman_gat_lstm.data.workbooks import (
    extract_bus_label_from_frequency_column,
    read_adjacency_workbook,
    read_frequency_workbook,
    read_koopman_workbook,
)


def test_extract_bus_label_from_frequency_column():
    assert extract_bus_label_from_frequency_column("1_母线电压相角/频率_BUS-39_BUS-39(Hz)") == "BUS39"


def test_extract_bus_label_from_frequency_column_rejects_embedded_labels():
    for column in ("XBUS39", "BUS39foo"):
        try:
            extract_bus_label_from_frequency_column(column)
        except ValueError as exc:
            assert "frequency column does not contain a BUS label" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {column}")


def test_read_frequency_workbook_returns_values_in_canonical_order(tmp_path):
    workbook_path = tmp_path / "frequency.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame(
            {
                "时间（秒）": [5.0, 5.01],
                "frequency_BUS-2(Hz)": [20.0, 21.0],
                "frequency_BUS-1(Hz)": [10.0, 11.0],
                "frequency_BUS-3(Hz)": [30.0, 31.0],
            }
        ).to_excel(writer, sheet_name="case-a", index=False)

    cases = read_frequency_workbook(workbook_path, ["BUS1", "BUS2", "BUS3"])

    assert list(cases) == ["case-a"]
    assert cases["case-a"]["node_labels"] == ["BUS1", "BUS2", "BUS3"]
    assert np.allclose(cases["case-a"]["timestamps"], np.array([5.0, 5.01]))
    assert np.allclose(cases["case-a"]["values"], np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]))


def test_read_koopman_workbook_returns_values_in_canonical_order(tmp_path):
    workbook_path = tmp_path / "koopman.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame(
            {
                "Dataset_Name": ["case-a_Koopman能量"],
                "BUS2": [2.0],
                "BUS1": [1.0],
                "BUS3": [3.0],
            }
        ).to_excel(writer, sheet_name="Sheet1", index=False)

    rows = read_koopman_workbook(workbook_path, ["BUS1", "BUS2", "BUS3"])

    assert list(rows) == ["case-a"]
    assert rows["case-a"]["node_labels"] == ["BUS1", "BUS2", "BUS3"]
    assert np.allclose(rows["case-a"]["values"], np.array([1.0, 2.0, 3.0]))


def test_read_koopman_workbook_detects_duplicate_normalized_case_ids(tmp_path):
    workbook_path = tmp_path / "koopman.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame(
            {
                "Dataset_Name": ["case-a_Koopman能量", "case-a"],
                "BUS1": [1.0, 4.0],
                "BUS2": [2.0, 5.0],
                "BUS3": [3.0, 6.0],
            }
        ).to_excel(writer, sheet_name="Sheet1", index=False)

    try:
        read_koopman_workbook(workbook_path, ["BUS1", "BUS2", "BUS3"])
    except ValueError as exc:
        assert "duplicate Koopman dataset name: case-a" in str(exc)
    else:
        raise AssertionError("expected duplicate normalized Koopman case ID")


def test_read_adjacency_workbook_returns_matrix_in_canonical_order(tmp_path):
    workbook_path = tmp_path / "adjacency.xlsx"
    adjacency = pd.DataFrame(
        [[22.0, 21.0, 23.0], [12.0, 11.0, 13.0], [32.0, 31.0, 33.0]],
        index=["BUS2", "BUS1", "BUS3"],
        columns=["BUS2", "BUS1", "BUS3"],
    )
    with pd.ExcelWriter(workbook_path) as writer:
        adjacency.to_excel(writer, sheet_name="原始邻接矩阵A")

    matrix, labels = read_adjacency_workbook(workbook_path, ["BUS1", "BUS2", "BUS3"])

    assert labels == ["BUS1", "BUS2", "BUS3"]
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, np.array([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0], [31.0, 32.0, 33.0]]))
