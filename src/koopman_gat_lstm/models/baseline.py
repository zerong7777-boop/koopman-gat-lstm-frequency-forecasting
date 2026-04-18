from torch import nn


class BaselineGATLSTM(nn.Module):
    """V1 baseline that keeps the model interface but uses no graph logic."""

    def __init__(
        self,
        num_nodes,
        input_feature_dim,
        gat_hidden_dim,
        gat_heads_layer1,
        gat_heads_layer2,
        lstm_hidden_dim,
        forecast_steps,
    ):
        super().__init__()
        if input_feature_dim != 1:
            raise ValueError("v1 baseline expects input_feature_dim=1")

        self.input_feature_dim = input_feature_dim
        self.num_nodes = num_nodes
        self.forecast_steps = forecast_steps
        self.lstm = nn.LSTM(
            input_size=num_nodes * input_feature_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_dim, forecast_steps * num_nodes)

    def forward(self, x, adj):
        """Accept raw frequency input as [batch, time, nodes] and ignore graph inputs."""
        del adj
        if x.ndim != 3:
            raise ValueError("expected x with shape [batch, time, nodes]")
        if x.shape[-1] != self.num_nodes:
            raise ValueError("input node dimension does not match num_nodes")

        x = x.reshape(x.shape[0], x.shape[1], self.num_nodes * self.input_feature_dim)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out.reshape(x.shape[0], self.forecast_steps, self.num_nodes)
