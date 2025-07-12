from torch import nn

from ..layers import MultiHeadAttention, OCRead

__all__ = ["BrainNetTF"]


class BrainNetTF(nn.Sequential):
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
        - As a result, the number of trainable parameters may not match those reported in the original paper.

    Args:
        num_embeddings (int): Input feature dimension per node (typically number of ROIs).
        num_outputs (int, optional): Number of outputs (e.g., 1 for regression, or C for classification).
            If `None`, omits the final flatten + linear layers. Default: `None`.
        num_hidden (int, optional): Hidden dimension used inside the attention layers.
            If `None`, defaults to `num_embeddings`. Default: `None`.
        num_heads (int): Number of attention heads per MHA layer. Default: 4.
        num_mha (int): Number of stacked MHA layers. Default: 1.
        num_clusters (int): Number of orthonormal clusters used in OCRead. Default: 10.
        bias (bool): Whether to include bias in linear layers. Default: `True`.
        device (torch.device, optional): Device on which the model's parameters will be allocated.
        dtype (torch.dtype, optional): Data type of the parameters.

    Input shape:
        - x: Tensor of shape `(batch_size, num_nodes, num_embeddings)`
          (e.g., a functional connectivity matrix where `num_nodes == num_embeddings`)

    Output shape:
        - Tensor of shape `(batch_size, num_outputs)`

    Example:
        >>> model = BrainNetTF(num_embeddings=200, num_outputs=1)
        >>> x = torch.randn(8, 200, 200)
        >>> x = (x + x.mT) / 2  # Symmetrize the input
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([8, 1])
    """

    def __init__(
        self,
        num_embeddings,
        num_outputs=None,
        num_hidden=None,
        num_heads=4,
        num_mha=1,
        num_clusters=10,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        num_hidden = num_hidden or num_embeddings

        # Add stacked multi-head attention layers
        for i in range(num_mha):
            self.add_module(
                f"mha_{i:0=2d}",
                MultiHeadAttention(num_embeddings, num_hidden, num_hidden, num_heads, bias, **factory_kwargs),
            )

        # Add OCRead for cluster-based pooling
        self.add_module("readout", OCRead(num_clusters, num_embeddings, **factory_kwargs))

        if num_outputs is not None:
            # Flatten cluster-wise features into a single vector
            self.add_module("flatten", nn.Flatten(start_dim=-2, end_dim=-1))

            # Final classification layer
            self.add_module("linear", nn.Linear(num_clusters * num_embeddings, num_outputs, bias, **factory_kwargs))
