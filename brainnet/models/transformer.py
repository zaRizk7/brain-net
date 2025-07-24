from torch import nn

from ..layers import ConcatLinear, MultiHeadAttention, OCRead, TFDecoder, TFEncoder

__all__ = ["BrainNetTF"]


class BrainNetTF(nn.Module):
    r"""
    Brain Network Transformer (BrainNetTF) model.

    This is a faithful PyTorch implementation of the architecture described in the
    BrainNetTF paper, with necessary clarifications and minimal implementation for reproducibility
    and understanding of the method.

    The model consists of:
        - A stack of Multi-Head Attention (MHA) layers to model pairwise relationships between ROIs
        - A clustering readout module (OCRead) that summarizes ROI features across orthonormal clusters
        - A flattening step to produce a single feature vector per subject
        - A final linear layer for classification or regression

    Notes:
        - The original BrainNetTF implementation includes additional undocumented components
          such as MLP classifiers and learnable node identity embeddings, which are omitted here.
        - The original paper used the full transformer encoder but the paper only describe using MHA layers.
            For extension, the `layer_type` argument can be used to specify whether to use MHA, encoder, or decoder layers.
        - As a result, the number of trainable parameters may not match those reported in the original paper.

    Args:
        num_inputs (int): Number of input features (e.g., number of ROIs).
        num_predictions (int, optional): Number of output predictions (e.g., number of classes).
            If `None`, no final linear layer is added. Default: `None`.
        num_layers (int): Number of MHA layers to stack. Default: 1.
        num_heads (int): Number of attention heads in each MHA layer. Default: 1.
        num_keys (int, optional): Number of keys for attention. If `None`, defaults to `num_inputs`. Default: `None`.
        num_values (int, optional):
            Number of values for attention. If `None`, defaults to `num_inputs`. Default: `None`.
        num_feedforward (int, optional):
            Number of feedforward units in the MHA layers. If `None`, defaults to `num_inputs`. Default: `None`.
        num_outputs (int, optional): Number of output features from the MHA layers.
            If `None`, defaults to `num_inputs`. Default: `None`.
        num_clusters (int): Number of orthonormal clusters for the readout layer. Default: 10.
        ortho_init_clusters (bool): If `True`, initializes clusters orthogonally. Default: `True`.
        learnable_clusters (bool): If `True`, clusters are learnable parameters. Default: `True`.
        p_dropout (float, optional): Dropout probability for MHA layers. If `None`, no dropout is applied. Default: `None`.
        bias (bool): Whether to include biases in MHA and linear layers. Default: `True`.
        norm_affine (bool): Whether to use affine normalization in MHA layers. Default: `True`.
        activation (str, optional): Activation function for MLP layers. If `None`, no activation is applied. Default: `None`.
        activation_mha (str, optional): Activation function for MHA layers. If `None`, softmax is used. Default: `None`.
        ln_eps (float): Epsilon value for LayerNorm in Transformer layers. Default: 1e-5.
        layer_type (str): Type of layer to use in the transformer stack.
            Supported types are 'mha', 'encoder', and 'decoder'. Default: 'mha'.
        device (torch.device, optional): Device to initialize model parameters on. Default: `None`.
        dtype (torch.dtype, optional): Data type for model parameters. Default: `None`.

    Examples:
        >>> model = BrainNetTF(num_inputs=200, num_predictions=1)
        >>> x = torch.randn(8, 200, 200)
        >>> x = (x + x.mT) / 2  # Symmetrize the input if needed
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([8, 1])
    """

    def __init__(
        self,
        num_inputs,
        num_predictions=None,
        num_layers=1,
        num_heads=1,
        num_keys=None,
        num_values=None,
        num_feedforward=None,
        num_outputs=None,
        num_clusters=10,
        ortho_init_clusters=True,
        learnable_clusters=True,
        p_dropout=None,
        bias=True,
        norm_affine=True,
        activation=None,
        activation_mha=None,
        ln_eps=1e-5,
        layer_type="mha",
        device=None,
        dtype=None,
    ):
        if layer_type not in {"mha", "encoder", "decoder"}:
            raise ValueError(f"Unsupported layer type: {layer_type}. Supported types are 'mha', 'encoder', 'decoder'.")
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        layer_kwargs = {
            "num_inputs": num_inputs,
            "num_heads": num_heads,
            "num_keys": num_keys,
            "num_values": num_values,
            "num_feedforward": num_feedforward,
            "num_outputs": num_outputs,
            "p_dropout": p_dropout,
            "bias": bias,
            "norm_affine": norm_affine,
            "activation": activation,
            "ln_eps": ln_eps,
            **factory_kwargs,
        }

        for i in range(num_layers):
            if layer_type == "mha":
                self.add_module(f"mha_{i:0=2d}", MultiHeadAttention(**layer_kwargs))
            elif layer_type == "encoder":
                self.add_module(f"encoder_{i:0=2d}", TFEncoder(**layer_kwargs))
            elif layer_type == "decoder":
                self.add_module(f"decoder_{i:0=2d}", TFDecoder(**layer_kwargs))

        # Readout layer for clustering
        self.readout = OCRead(num_clusters, num_inputs, ortho_init_clusters, learnable_clusters, **factory_kwargs)

        if num_predictions is not None:
            # Final linear layer to produce the output
            self.add_module("linear", ConcatLinear(num_inputs, num_predictions, num_clusters, bias, **factory_kwargs))
        else:
            self.add_module("linear", nn.Identity())

    def forward(
        self,
        x,
        z=None,
        return_attention=False,
        return_assignments=False,
        s_mask=None,
        c_mask=None,
        s_causal_mask=False,
        c_causal_mask=False,
    ):
        """
        Forward pass through the BrainNetTF model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, num_embeddings).
            z (torch.Tensor, optional): Optional tensor for cross-attention. Default: `None`.
            return_attention (bool): If `True`, returns attention weights. Default: `False`.
            return_assignments (bool): If `True`, returns the assignment weights from the readout layer.
                Default: `False`.
            s_mask (torch.Tensor, optional): Source mask for attention. Default: `None`.
            c_mask (torch.Tensor, optional): Cross mask for attention. Default: `None`.
            s_causal_mask (bool): If `True`, applies causal masking to source attention. Default: `False`.
            c_causal_mask (bool): If `True`, applies causal masking to cross attention. Default: `False`.

        Returns:
            torch.Tensor: Output tensor after processing through MHA layers and readout.
        """
        s_attn_maps, c_attn_maps = [], []
        for name, module in self.named_children():
            if "mha" in name:
                x = module(x, z, return_attention, s_mask, s_causal_mask)
            elif "encoder" in name:
                x = module(x, z, return_attention, s_mask, c_mask)
            elif "decoder" in name:
                x = module(x, z, return_attention, s_mask, c_mask, s_causal_mask, c_causal_mask)
            else:
                break

            if return_attention and "decoder" in name:
                x, s_attn, c_attn = x
                s_attn_maps.append((s_attn))
                c_attn_maps.append((c_attn))
            elif return_attention:
                x, s_attn = x
                s_attn_maps.append(s_attn)

        # Apply readout layer
        x = self.readout(x, return_assignments)
        if return_assignments:
            x, assignments = x

        # Final linear layer if specified
        x = self.linear(x)

        if return_attention and return_assignments:
            return x, s_attn_maps, c_attn_maps, assignments
        elif return_attention:
            return x, s_attn_maps, c_attn_maps
        elif return_assignments:
            return x, assignments
        return x


if __name__ == "__main__":
    import torch

    batch = 8
    nodes = 360
    embeddings = 360

    # Create a random input tensor (e.g., functional connectivity matrix)
    x = torch.randn(batch, nodes, embeddings)
    x = (x + x.mT) / 2  # Symmetrize the input

    # Initialize and run the model
    model = BrainNetTF(360, num_layers=2, num_heads=4, num_feedforward=2048, layer_type="encoder")
    dim_reduction = nn.Sequential(nn.Linear(360, 8), nn.LeakyReLU())
    fc = nn.Sequential(
        ConcatLinear(8, 256, 10),
        nn.LeakyReLU(),
        nn.Linear(256, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 2),
    )
    out = model(x)
    print(out.shape)  # Expected output shape: (batch_size, num_outputs)
    print(model)
    # Total parameters with dim_reduction and fc
    total_params = (
        sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in dim_reduction.parameters())
        + sum(p.numel() for p in fc.parameters())
    )
    print(f"Total parameters: {total_params}")

    print(model.readout)
