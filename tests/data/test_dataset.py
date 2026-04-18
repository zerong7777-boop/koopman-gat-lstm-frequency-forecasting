import numpy as np

from koopman_gat_lstm.data.dataset import slice_supervised_sample
from koopman_gat_lstm.data.normalization import (
    apply_frequency_standardization,
    fit_frequency_stats,
    normalize_koopman,
)


def test_slice_supervised_sample_uses_half_open_time_windows():
    timestamps = np.round(np.arange(5.00, 8.01, 0.01), 2)
    values = np.stack([timestamps, timestamps + 1], axis=1)

    x, y = slice_supervised_sample(
        timestamps=timestamps,
        node_values=values,
        input_start=5.00,
        input_end=5.40,
        forecast_start=5.40,
        forecast_end=8.00,
    )

    assert x.shape == (40, 2)
    assert y.shape == (260, 2)
    assert x[0, 0] == 5.00
    assert x[-1, 0] == 5.39
    assert y[0, 0] == 5.40
    assert y[-1, 0] == 7.99


def test_slice_supervised_sample_uses_rounded_timestamps_for_boundary_membership():
    timestamps = np.round(np.arange(5.00, 8.01, 0.01), 2)
    timestamps[40] = 5.399999999
    values = np.stack([timestamps, timestamps + 1], axis=1)

    x, y = slice_supervised_sample(
        timestamps=timestamps,
        node_values=values,
        input_start=5.00,
        input_end=5.40,
        forecast_start=5.40,
        forecast_end=8.00,
    )

    assert x.shape == (40, 2)
    assert y.shape == (260, 2)
    assert x[-1, 0] == 5.39
    assert y[0, 0] == 5.399999999


def test_slice_supervised_sample_rejects_unsorted_rounded_timestamps():
    timestamps = np.round(np.arange(5.00, 8.01, 0.01), 2)
    timestamps[10], timestamps[11] = timestamps[11], timestamps[10]
    values = np.stack([timestamps, timestamps + 1], axis=1)

    try:
        slice_supervised_sample(
            timestamps=timestamps,
            node_values=values,
            input_start=5.00,
            input_end=5.40,
            forecast_start=5.40,
            forecast_end=8.00,
        )
    except ValueError as exc:
        assert "sorted ascending" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsorted rounded timestamps")


def test_frequency_stats_support_per_node_standardization_application():
    train_x = np.array([[[1.0, 10.0], [3.0, 14.0]]])
    train_y = np.array([[[5.0, 18.0], [7.0, 22.0]]])
    stats = fit_frequency_stats(train_x, train_y)
    values = np.array([[[4.0, 16.0], [6.23606798, 20.47213595]]])
    standardized = apply_frequency_standardization(values, stats)

    assert np.allclose(stats["mean"], np.array([4.0, 16.0]))
    assert np.allclose(stats["std"], np.array([2.23606798, 4.47213595]))
    assert np.allclose(standardized, np.array([[[0.0, 0.0], [1.0, 1.0]]]))


def test_normalize_koopman_scales_each_sample_over_its_node_dimension():
    koopman = normalize_koopman(np.array([[2.0, 8.0], [5.0, 15.0]]))

    assert np.allclose(koopman, np.array([[0.0, 1.0], [0.0, 1.0]]))


def test_normalize_koopman_returns_zeros_for_constant_rows():
    koopman = normalize_koopman(np.array([[3.0, 3.0], [1.0, 5.0]]))

    assert np.allclose(koopman, np.array([[0.0, 0.0], [0.0, 1.0]]))
