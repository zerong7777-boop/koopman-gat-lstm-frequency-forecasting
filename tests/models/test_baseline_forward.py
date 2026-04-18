import pytest
import torch

from koopman_gat_lstm.models.baseline import BaselineGATLSTM


def test_baseline_forward_returns_expected_shape():
    model = BaselineGATLSTM(
        num_nodes=39,
        input_feature_dim=1,
        gat_hidden_dim=8,
        gat_heads_layer1=2,
        gat_heads_layer2=2,
        lstm_hidden_dim=16,
        forecast_steps=260,
    )
    x = torch.arange(4 * 40 * 39, dtype=torch.float32).reshape(4, 40, 39)
    adj = torch.ones(39, 39)
    lstm_inputs = []

    def capture_lstm_input(module, args):
        del module
        lstm_inputs.append(args[0].detach().clone())

    hook = model.lstm.register_forward_pre_hook(capture_lstm_input)

    assert model.lstm.input_size == 39

    try:
        y = model(x, adj)
    finally:
        hook.remove()

    expected_lstm_input = x.unsqueeze(-1).reshape(4, 40, 39)

    assert len(lstm_inputs) == 1
    assert torch.equal(lstm_inputs[0], expected_lstm_input)

    assert y.shape == (4, 260, 39)


def test_baseline_forward_rejects_malformed_input_shape():
    model = BaselineGATLSTM(
        num_nodes=39,
        input_feature_dim=1,
        gat_hidden_dim=8,
        gat_heads_layer1=2,
        gat_heads_layer2=2,
        lstm_hidden_dim=16,
        forecast_steps=260,
    )

    bad_x = torch.randn(4, 40, 13, 3)
    adj = torch.ones(39, 39)

    with pytest.raises(ValueError, match="expected x with shape \\[batch, time, nodes\\]"):
        model(bad_x, adj)


def test_baseline_forward_rejects_wrong_node_dimension():
    model = BaselineGATLSTM(
        num_nodes=39,
        input_feature_dim=1,
        gat_hidden_dim=8,
        gat_heads_layer1=2,
        gat_heads_layer2=2,
        lstm_hidden_dim=16,
        forecast_steps=260,
    )

    bad_x = torch.randn(4, 40, 38)
    adj = torch.ones(39, 39)

    with pytest.raises(ValueError, match="input node dimension does not match num_nodes"):
        model(bad_x, adj)


def test_baseline_forward_rejects_non_v1_input_feature_dim():
    with pytest.raises(ValueError, match="v1 baseline expects input_feature_dim=1"):
        BaselineGATLSTM(
            num_nodes=39,
            input_feature_dim=2,
            gat_hidden_dim=8,
            gat_heads_layer1=2,
            gat_heads_layer2=2,
            lstm_hidden_dim=16,
            forecast_steps=260,
        )
