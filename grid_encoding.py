import tinycudann as tc
import torch
import torch.nn as nn


def get_a_grid_encoder(
    input_dims=3,
    L=16,
    F=2,
    log2_T=19,
    N_min=16,
    scale=1.3819,
    interpolation="Smoothstep",
):
    encoder = tc.Encoding(
        n_input_dims=input_dims,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": L,
            "n_features_per_level": F,
            "log2_hashmap_size": log2_T,
            "base_resolution": N_min,
            "per_level_scale": scale,
            "interpolation": interpolation,
        },
    )
    return encoder


def get_fully_fused_mlp(
    n_input_dims=2,
    n_output_dims=128,
    activation="Tanh",
    n_neurons=64,
    output_activation="None",
    n_hidden_layers=3,
    network_config=None,
):
    if network_config is None:
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": activation,
            "output_activation": output_activation,
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
        }
    mlp = tc.Network(
        n_input_dims=n_input_dims,
        n_output_dims=n_output_dims,
        network_config=network_config,
    )
    return mlp


def get_freq_encoder(n_input_dims=1, num_freqs=32):
    return tc.Encoding(
        n_input_dims=n_input_dims,
        encoding_config={
            "otype": "Frequency",
            "n_frequencies": num_freqs,
        },
    )


class SDFNetwork(nn.Module):
    def __init__(
        self,
        encoding="hashgrid",
        num_layers=3,
        skips=[],
        hidden_dim=64,
        clip_sdf=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf

        assert (
            self.skips == []
        ), "tc does not support concatenating inside, please use skips=[]."

        self.encoder = tc.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
        )

        self.backbone = tc.Network(
            n_input_dims=32,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

    def forward(self, x):
        # x: [B, 3]

        x = (x + 1) / 2  # to [0, 1]
        x = self.encoder(x)
        h = self.backbone(x)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h
