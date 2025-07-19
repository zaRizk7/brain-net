import torch
from torch import nn

__all__ = ["Attention"]


class Attention(nn.Module):
    r"""
    Scaled dot-product attention with optional causal masking.

    This module computes the scaled dot-product attention, optionally applying a causal mask to prevent
    attending to future positions.

    Args:
        activation (nn.Module, optional): Activation function to apply to attention scores. Defaults: `nn.Softmax(dim=-1)`.

    Shape:
        - Inputs: Tuple of (Q, K, V), each with shape (..., seq_len, dim)
        - Output: Tensor of shape (..., seq_len, dim)
        - Attention weights: Tensor of shape (..., seq_len, seq_len)

    Example:
        >>> attn = Attention(mask=True)
        >>> q = torch.randn(2, 4, 10, 64)  # Batch of 2, num_heads=4, seq_len=10, dim=64
        >>> k = torch.randn(2, 4, 10, 64)
        >>> v = torch.randn(2, 4, 10, 64)
        >>> output, attn_weights = attn((q, k, v))
        >>> print(output.shape)  # Expected: (2, 4, 10, 64)
        >>> print(attn_weights.shape)  # Expected: (2, 4, 10, 10)
    """

    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation or nn.Softmax(dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(activation={self.activation})"

    def forward(self, qkv, mask=None, causal_mask=False):
        r"""
        Args:
            qkv (tuple of torch.Tensor): Tuple containing query, key, and value tensors, each of shape
                (..., seq_len, dim).
            mask (torch.Tensor, optional): Optional mask tensor of shape (..., seq_len, seq_len) to apply to the
                attention scores.
            causal_mask (bool, optional): If True, applies a causal mask to prevent attending to future positions.
                Overrides the `mask` argument if set to True.

        Returns:
            torch.Tensor: Output tensor after attention of shape (..., seq_len, dim).
            torch.Tensor: Attention weights of shape (..., seq_len, seq_len).
        """
        q, k, v = qkv
        d_k = k.size(-1)

        # Compute scaled attention scores
        scores = torch.matmul(q, k.mT) / (d_k**0.5)

        # Apply causal mask if enabled (assuming q and k have same seq_len)
        if causal_mask:
            seq_len = q.size(-2)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            scores = scores + mask * float("-inf")
        elif mask:
            scores = scores * mask

        # Apply softmax and attend to values
        attn = self.activation(scores)
        z = torch.matmul(attn, v)

        return z, attn
