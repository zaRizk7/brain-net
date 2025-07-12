from torch import nn

from ..layers import ConcatLinear, MultiHeadAttention, OCRead

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
        num_embeddings (int): Input feature dimension per node (typically equal to number of ROIs).
        num_outputs (int, optional): Number of outputs (e.g., 1 for regression or C for classification).
            If `None`, the final flatten + linear layers are omitted. Default: `None`.
        num_key (int, optional): Dimension of the key vectors in attention. Defaults to `num_embeddings`.
        num_value (int, optional): Dimension of the value vectors in attention. Defaults to `num_embeddings`.
        num_heads (int): Number of attention heads in each MHA layer. Default: 4.
        num_mha (int): Number of stacked MHA layers. Default: 1.
        num_clusters (int): Number of orthonormal clusters used in OCRead. Default: 10.
        bias (bool): Whether to include bias terms in linear layers. Default: `True`.
        device (torch.device, optional): Device to initialize model parameters on.
        dtype (torch.dtype, optional): Data type for model parameters.

    Examples:
        >>> model = BrainNetTF(num_embeddings=200, num_outputs=1)
        >>> x = torch.randn(8, 200, 200)
        >>> x = (x + x.mT) / 2  # Symmetrize the input if needed
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([8, 1])
    """

    def __init__(
        self,
        num_embeddings,
        num_outputs=None,
        num_key=None,
        num_value=None,
        num_heads=4,
        num_mha=1,
        num_clusters=10,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        num_key = num_key or num_embeddings
        num_value = num_value or num_embeddings

        # Add stacked multi-head attention layers
        for i in range(num_mha):
            self.add_module(
                f"mha_{i:0=2d}",
                MultiHeadAttention(num_embeddings, num_key, num_value, num_heads, bias, **factory_kwargs),
            )

        # Add OCRead for cluster-based pooling
        self.add_module("readout", OCRead(num_clusters, num_embeddings, **factory_kwargs))

        if num_outputs is not None:
            # Final classification layer
            self.add_module("linear", ConcatLinear(num_embeddings, num_outputs, num_clusters, bias, **factory_kwargs))
