import torch

from koopman_gat_lstm.models.gat_layers import (
    DenseGraphAttentionLayer,
    masked_attention_softmax,
)
from koopman_gat_lstm.models.koopman import (
    NEAR_NEUTRAL_BETA_INIT,
    KoopmanGATLSTM,
    apply_koopman_logit_correction,
)


def test_koopman_logit_correction_matches_softplus_beta_source_prior_formula():
    logits = torch.zeros(1, 2, 2, 3)
    koopman = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    beta = torch.tensor([0.0, 1.0], dtype=torch.float32)

    corrected = apply_koopman_logit_correction(logits, koopman, beta)

    expected = logits + torch.nn.functional.softplus(beta).view(1, 2, 1, 1) * koopman.view(1, 1, 1, 3)

    assert corrected.shape == logits.shape
    assert torch.allclose(corrected, expected)
    assert torch.all(corrected[:, 0, :, 2] > corrected[:, 0, :, 0])
    assert torch.all(corrected[:, 1, :, 2] > corrected[:, 1, :, 1])


def test_koopman_model_variant_preserves_v1_forecast_contract():
    torch.manual_seed(7)
    model = KoopmanGATLSTM(
        num_nodes=39,
        input_feature_dim=1,
        gat_hidden_dim=8,
        gat_heads_layer1=2,
        gat_heads_layer2=3,
        lstm_hidden_dim=16,
        forecast_steps=260,
    )
    x = torch.randn(4, 40, 39)
    adj = torch.ones(39, 39)
    koopman = torch.rand(4, 39)

    y = model(x, adj, koopman)

    assert model.beta_layer1.shape == (2,)
    assert model.beta_layer2.shape == (3,)
    assert torch.allclose(model.beta_layer1.detach(), torch.full((2,), NEAR_NEUTRAL_BETA_INIT))
    assert torch.allclose(model.beta_layer2.detach(), torch.full((3,), NEAR_NEUTRAL_BETA_INIT))
    assert y.shape == (4, 260, 39)


def test_koopman_model_changes_attention_when_koopman_prior_changes():
    torch.manual_seed(11)
    model = KoopmanGATLSTM(
        num_nodes=3,
        input_feature_dim=1,
        gat_hidden_dim=4,
        gat_heads_layer1=2,
        gat_heads_layer2=2,
        lstm_hidden_dim=5,
        forecast_steps=6,
    )
    x = torch.ones(1, 2, 3)
    adj = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    koopman_low = torch.zeros(1, 3)
    koopman_high = torch.tensor([[0.0, 0.0, 4.0]])

    with torch.no_grad():
        model.beta_layer1.fill_(2.0)
        model.beta_layer2.fill_(2.0)

    _, attention_low = model(x, adj, koopman_low, return_attention=True)
    _, attention_high = model(x, adj, koopman_high, return_attention=True)

    assert not torch.allclose(attention_low["layer1"], attention_high["layer1"])
    assert not torch.allclose(attention_low["layer2"], attention_high["layer2"])


def test_dense_attention_keeps_masked_columns_and_isolated_rows_at_zero():
    torch.manual_seed(3)
    layer = DenseGraphAttentionLayer(
        input_dim=1,
        output_dim=2,
        num_heads=1,
        concat=False,
    )
    x = torch.tensor([[[1.0], [2.0], [3.0]]])
    adj = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    source_prior = torch.tensor([[0.1, 0.2, 0.3]])
    beta = torch.tensor([1.0])

    _, attention = layer(
        x,
        adj,
        source_prior=source_prior,
        beta=beta,
        return_attention=True,
    )

    assert torch.allclose(attention[0, 0, 0, 1:], torch.zeros(2))
    assert torch.allclose(attention[0, 0, 1], torch.zeros(3))
    assert torch.isclose(attention[0, 0, 1].sum(), torch.tensor(0.0))


def test_masked_attention_softmax_zeroes_forbidden_columns():
    logits = torch.tensor([[[[1.0, 4.0, -2.0]]]])
    mask = torch.tensor([[[[True, False, True]]]])

    attention = masked_attention_softmax(logits, mask)

    assert torch.isclose(attention[0, 0, 0, 1], torch.tensor(0.0))
    assert torch.isclose(attention[0, 0, 0].sum(), torch.tensor(1.0))


def test_masked_attention_softmax_keeps_fully_masked_rows_at_zero():
    logits = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask = torch.tensor([[[[True, False], [False, False]]]])

    attention = masked_attention_softmax(logits, mask)

    assert torch.allclose(attention[0, 0, 1], torch.zeros(2))
    assert torch.isclose(attention[0, 0, 1].sum(), torch.tensor(0.0))


def test_masked_attention_softmax_stays_finite_for_large_negative_logits():
    logits = torch.tensor([[[[-1000.0, -1200.0, 0.0]]]])
    mask = torch.tensor([[[[True, True, False]]]])

    attention = masked_attention_softmax(logits, mask)

    assert torch.isfinite(attention).all()
    assert torch.isclose(attention[0, 0, 0, 2], torch.tensor(0.0))
