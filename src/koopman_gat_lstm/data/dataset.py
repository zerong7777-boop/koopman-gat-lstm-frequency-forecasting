import numpy as np


def _expected_window_steps(start, end, sample_rate_hz):
    expected = (end - start) * sample_rate_hz
    rounded = round(expected)
    if not np.isclose(expected, rounded):
        raise ValueError("window bounds do not align to the sample rate")
    return int(rounded)


def slice_supervised_sample(
    timestamps,
    node_values,
    input_start,
    input_end,
    forecast_start,
    forecast_end,
    sample_rate_hz=100,
):
    timestamps = np.asarray(timestamps, dtype=float)
    node_values = np.asarray(node_values)
    if timestamps.shape[0] != node_values.shape[0]:
        raise ValueError("timestamps and node_values must have the same row count")

    rounded_timestamps = np.round(timestamps, 2)
    if np.any(np.diff(rounded_timestamps) <= 0):
        raise ValueError("timestamps must be sorted ascending on the rounded 0.01s grid")

    input_mask = (rounded_timestamps >= input_start) & (rounded_timestamps < input_end)
    forecast_mask = (rounded_timestamps >= forecast_start) & (rounded_timestamps < forecast_end)
    x = node_values[input_mask]
    y = node_values[forecast_mask]
    expected_input_steps = _expected_window_steps(input_start, input_end, sample_rate_hz)
    expected_forecast_steps = _expected_window_steps(forecast_start, forecast_end, sample_rate_hz)
    step = 1.0 / sample_rate_hz
    expected_input_grid = np.round(input_start + np.arange(expected_input_steps) * step, 2)
    expected_forecast_grid = np.round(forecast_start + np.arange(expected_forecast_steps) * step, 2)
    if x.shape[0] != expected_input_steps or y.shape[0] != expected_forecast_steps:
        raise ValueError("invalid supervised window")
    if not np.array_equal(rounded_timestamps[input_mask], expected_input_grid):
        raise ValueError("input window does not match the expected rounded timestamp grid")
    if not np.array_equal(rounded_timestamps[forecast_mask], expected_forecast_grid):
        raise ValueError("forecast window does not match the expected rounded timestamp grid")
    return x, y
