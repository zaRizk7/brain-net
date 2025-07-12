import torch
from torch import nn

from .attention import Attention

__all__ = ["MultiHeadLinear", "ConcatLinear", "MultiHeadAttention"]


class MultiHeadLinear(nn.Module):
    r"""
    Applies multiple independent linear transformations (one per head) to the last input dimension.

    Each head learns a separate linear projection:
    - input shape: `(..., num_inputs)`
    - output shape: `(..., num_heads, num_outputs)`

    Args:
        num_inputs (int): Size of each input sample.
        num_outputs (int): Size of each output sample.
        num_heads (int): Number of independent projection heads. Default: 1
        bias (bool): If set to `False`, the layer will not learn an additive bias. Default: `True`
        device (torch.device, optional): The device for the parameters.
        dtype (torch.dtype, optional): The data type for the parameters.

    Shape:
        - Input: `(..., num_inputs)`
        - Output: `(..., num_heads, num_outputs)`
    """

    __constants__ = ["num_inputs", "num_outputs", "num_heads"]
    num_inputs: int
    num_outputs: int
    num_heads: int
    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, num_inputs, num_outputs, num_heads=1, bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_heads = num_heads

        self.weight = nn.Parameter(torch.empty(num_heads, num_outputs, num_inputs, **factory_kwargs))
        self.init_bias(bias, **factory_kwargs)
        self.reset_parameters()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_inputs={self.num_inputs}, "
            f"num_outputs={self.num_outputs}, num_heads={self.num_heads}, "
            f"bias={self.bias is not None})"
        )

    def init_bias(self, bias, **kwargs):
        """
        Initializes the bias term for each head.

        Args:
            bias (bool): Whether to include bias.
            **kwargs: Additional keyword arguments for tensor creation (e.g., device, dtype).
        """
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_heads, self.num_outputs, **kwargs))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        """
        Initializes weights using Xavier uniform initialization for each head.
        Bias is initialized to zero.
        """
        for i in range(self.num_heads):
            nn.init.xavier_uniform_(self.weight[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape `(..., num_inputs)`

        Returns:
            Tensor: Output tensor of shape `(..., num_heads, num_outputs)`
        """
        bias = self.bias if self.bias is not None else 0
        return torch.einsum("...i,hji->...hj", x, self.weight) + bias


class ConcatLinear(MultiHeadLinear):
    r"""
    Applies multiple independent linear transformations (one per head), then concatenates the results.

    Compared to `MultiHeadLinear`, this class concatenates the inputs first before applying linear
    transformation.

    Shape:
        - Input: `(..., num_heads, num_inputs)`
        - Output: `(..., num_outputs)`
    """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_inputs={self.num_inputs}, "
            f"num_outputs={self.num_outputs}, num_heads={self.num_heads}, "
            f"bias={self.bias is not None})"
        )

    def init_bias(self, bias, **kwargs):
        """
        Initializes a single shared bias term for concatenated output.

        Args:
            bias (bool): Whether to include bias.
            **kwargs: Additional keyword arguments for tensor creation.
        """
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_outputs, **kwargs))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        """
        Initializes weights using Xavier uniform initialization for by concatenated heads.
        The weights are reshaped to match the concatenated output.
        Bias is initialized to zero.
        """
        weight = self.weight.data.view(self.num_outputs, -1)
        nn.init.xavier_uniform_(weight)
        self.weight.data = weight.view(self.num_heads, self.num_outputs, self.num_inputs)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape `(..., num_heads, num_inputs)`

        Returns:
            Tensor: Output tensor of shape `(..., num_outputs)`
        """
        bias = self.bias if self.bias is not None else 0
        return torch.einsum("...hi,hji->...j", x, self.weight) + bias


class MultiHeadAttention(nn.Module):
    r"""
    Multi-head attention block using separate linear projections per head.

    This module computes multi-head queries, keys, values using `MultiHeadLinear`,
    applies attention, and combines the heads using `ConcatLinear`.

    Args:
        d_model (int): Input feature dimensionality.
        d_k (int): Dimension of the query/key vectors.
        d_v (int): Dimension of the value vectors.
        num_heads (int): Number of attention heads. Default: 1
        bias (bool): Whether to include bias in linear projections. Default: `True`
        mask (bool): Whether to apply causal masking in attention. Default: `False`
        device (torch.device, optional): The device for the parameters.
        dtype (torch.dtype, optional): The data type for the parameters.

    Shape:
        - Input: `(..., d_model)`
        - Output: `(..., d_o)`
    """

    def __init__(self, d_model, d_k, d_v, num_heads=1, bias=True, mask=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.linear_q = MultiHeadLinear(d_model, d_k, num_heads, bias, **factory_kwargs)
        self.linear_k = MultiHeadLinear(d_model, d_k, num_heads, bias, **factory_kwargs)
        self.linear_v = MultiHeadLinear(d_model, d_v, num_heads, bias, **factory_kwargs)
        self.linear_o = ConcatLinear(d_v, d_model, num_heads, bias, **factory_kwargs)
        self.attention = Attention(mask)

    def forward(self, x, return_attention=False):
        """
        Args:
            x (Tensor): Input tensor of shape `(..., d_model)`

        Returns:
            Tensor: Output tensor of shape `(..., d_o)`
        """
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        z, attn = self.attention((q, k, v))
        z = self.linear_o(z)

        if not return_attention:
            return z
        return z, attn
