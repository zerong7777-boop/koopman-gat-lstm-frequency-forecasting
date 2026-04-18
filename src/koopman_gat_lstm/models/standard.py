from __future__ import annotations

import torch
from torch import nn

from koopman_gat_lstm.models.gat_layers import DenseGraphAttentionLayer


class StandardGATLSTM(nn.Module):
    """Standard two-layer GAT-LSTM forecast baseline without Koopman priors."""

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
            raise ValueError("v1 standard model expects input_feature_dim=1")

        self.num_nodes = num_nodes
        self.input_feature_dim = input_feature_dim
        self.gat_hidden_dim = gat_hidden_dim
        self.forecast_steps = forecast_steps
        self.gat_layer1 = DenseGraphAttentionLayer(
            input_dim=input_feature_dim,
            output_dim=gat_hidden_dim,
            num_heads=gat_heads_layer1,
            concat=True,
        )
        self.gat_layer2 = DenseGraphAttentionLayer(
            input_dim=gat_hidden_dim * gat_heads_layer1,
            output_dim=gat_hidden_dim,
            num_heads=gat_heads_layer2,
            concat=False,
        )
        self.activation = nn.ELU()
        self.lstm = nn.LSTM(
            input_size=num_nodes * gat_hidden_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_dim, forecast_steps * num_nodes)

    def forward(self, x, adj, return_attention: bool = False):
        if x.ndim != 3:
            raise ValueError("expected x with shape [batch, time, nodes]")
        if x.shape[-1] != self.num_nodes:
            raise ValueError("input node dimension does not match num_nodes")
        if adj.ndim != 2:
            raise ValueError("expected adj with shape [nodes, nodes]")
        if adj.shape[0] != self.num_nodes or adj.shape[1] != self.num_nodes:
            raise ValueError("adjacency shape does not match num_nodes")

        batch_size, time_steps, _ = x.shape
        timestep_inputs = x.unsqueeze(-1).reshape(batch_size * time_steps, self.num_nodes, self.input_feature_dim)

        layer1_result = self.gat_layer1(
            timestep_inputs,
            adj,
            return_attention=return_attention,
        )
        if return_attention:
            layer1_out, attn1 = layer1_result
        else:
            layer1_out = layer1_result
        layer1_out = self.activation(layer1_out)
        layer2_result = self.gat_layer2(
            layer1_out,
            adj,
            return_attention=return_attention,
        )
        if return_attention:
            layer2_out, attn2 = layer2_result
        else:
            layer2_out = layer2_result
        layer2_out = self.activation(layer2_out)

        lstm_inputs = layer2_out.reshape(batch_size, time_steps, self.num_nodes * self.gat_hidden_dim)
        _, (hidden, _) = self.lstm(lstm_inputs)
        out = self.fc(hidden[-1]).reshape(batch_size, self.forecast_steps, self.num_nodes)

        if return_attention:
            attention = {
                "layer1": attn1.reshape(batch_size, time_steps, attn1.shape[1], self.num_nodes, self.num_nodes),
                "layer2": attn2.reshape(batch_size, time_steps, attn2.shape[1], self.num_nodes, self.num_nodes),
            }
            return out, attention
        return out
