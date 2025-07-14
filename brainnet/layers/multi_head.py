import torch
from torch import nn

from .attention import Attention

__all__ = ["MultiHeadLinear", "ConcatLinear", "MultiHeadAttention"]


class MultiHeadLinear(nn.Module):
    r"""
    Applies multiple independent linear transformations (one per head) to the last input dimension.

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

    Variables:
        - `weight`: Learnable weights of shape `(num_heads, num_outputs, num_inputs)`.
        - `bias`: Learnable bias of shape `(num_heads, num_outputs)` if `bias` is `True`, otherwise `None`.

    Examples:
        >>> x = torch.randn(10, 64)  # Batch of 10, 64 inputs
        >>> linear = MultiHeadLinear(64, 128, num_heads=4)
        >>> output = linear(x)  # Output shape `(10, 4, 128)`
        >>> print(output.shape)  # torch.Size([10, 4, 128])
        >>> print(linear)  # MultiHeadLinear(num_inputs=64, num_outputs=128, num_heads=4, bias=True)
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
    Aggregate inputs from multiple heads then apply linear transformation.

    Compared to `MultiHeadLinear`, this class concatenates the inputs first before applying linear
    transformation.

    Args:
        num_inputs (int): Size of each input sample.
        num_outputs (int): Size of each output sample.
        num_heads (int): Number of independent projection heads. Default: 1
        bias (bool): If set to `False`, the layer will not learn an additive bias. Default: `True`
        device (torch.device, optional): The device for the parameters.
        dtype (torch.dtype, optional): The data type for the parameters.

    Shape:
        - Input: `(..., num_heads, num_inputs)`
        - Output: `(..., num_outputs)`

    Variables:
        - `weight`: Learnable weights of shape `(num_heads, num_outputs, num_inputs)`.
        - `bias`: Learnable bias of shape `(num_outputs,)` if `bias` is `True`, otherwise `None`.

    Examples:
        >>> x = torch.randn(10, 4, 64)  # Batch of 10, 4 heads, 64 inputs
        >>> linear = ConcatLinear(64, 128, num_heads=4)
        >>> output = linear(x)  # Output shape `(10, 128)`
        >>> print(output.shape)  # torch.Size([10, 128])
        >>> print(linear)  # ConcatLinear(num_inputs=64, num_outputs=128, num_heads=4, bias=True)
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
        - Input: `(..., seq_len, d_model)`
        - Output: `(..., seq_len, d_model)`
        - Attention weights: `(..., num_heads, seq_len, seq_len)` if `return_attention=True`

    Examples:
        >>> mha = MultiHeadAttention(d_model=512, d_k=64, d_v=64, num_heads=8)
        >>> x = torch.randn(10, 20, 512)  # Batch of 10, sequence length of 20, feature size of 512
        >>> output, attn_weights = mha(x, return_attention=True)
        >>> print(output.shape)  # Expected: (10, 20, 512)
        >>> print(attn_weights.shape)  # Expected: (10, 8, 20, 20)
    """

    def __init__(self, d_model, d_k, d_v, num_heads=1, bias=True, mask=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.linear_q = MultiHeadLinear(d_model, d_k, num_heads, bias, **factory_kwargs)
        self.linear_k = MultiHeadLinear(d_model, d_k, num_heads, bias, **factory_kwargs)
        self.linear_v = MultiHeadLinear(d_model, d_v, num_heads, bias, **factory_kwargs)
        self.linear_o = ConcatLinear(d_v, d_model, num_heads, bias, **factory_kwargs)
        self.attention = Attention(mask)

    def forward(self, x, z=None, return_attention=False):
        """
        Args:
            x (Tensor): Input tensor of shape `(..., d_model)`
            z (Tensor, optional): Optional tensor for cross-attention. If provided,
                it should have the same last dimension as `x`. Default: `None`.
            return_attention (bool): If `True`, also return attention weights. Default: `False`.

        Returns:
            Tensor: Output tensor of shape `(..., d_o)`
            Tensor (optional): Attention weights of shape `(..., num_heads, seq_len, seq_len)` if `return_attention` is `True`.
        """
        z = x if z is None else z
        # Apply multi-head linear projections
        # Shape: (batch_size, seq_len, num_heads, d_k) for Q
        # Shape: (batch_size, seq_len, num_heads, d_k) for K
        # Shape: (batch_size, seq_len, num_heads, d_v) for V
        q, k, v = self.linear_q(x), self.linear_k(z), self.linear_v(z)
        # Transpose to get shape (batch_size, num_heads, seq_len, dim)
        q, k, v = q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2)

        z, attn = self.attention((q, k, v))
        # Apply aggregation and linear projection
        # Shape: (batch_size, seq_len, num_heads, d_v) -> (batch_size, seq_len, d_model)
        z = self.linear_o(z.transpose(-3, -2))

        return (z, attn) if return_attention else z
