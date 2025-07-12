import torch
from torch import nn

__all__ = ["Attention"]


class Attention(nn.Module):
    r"""
    Scaled dot-product attention with optional causal masking.

    This implements:
        Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k) + mask) @ V

    Args:
        mask (bool): Whether to apply causal (upper-triangular) masking to the attention scores.

    Shape:
        - Input: `(batch_size, seq_len, num_heads, dim)` for Q, K, V
        - Output: `(batch_size, seq_len, num_heads, dim)`
        - Attention weights: `(batch_size, num_heads, seq_len, seq_len)`

    Example:
        >>> attn = Attention(mask=True)
        >>> q = torch.randn(2, 10, 4, 64)  # Batch of 2, seq_len=10, num_heads=4, dim=64
        >>> k = torch.randn(2, 10, 4, 64)
        >>> v = torch.randn(2, 10, 4, 64)
        >>> output, attn_weights = attn((q, k, v))
        >>> print(output.shape)  # Expected: (2, 10, 4, 64)
        >>> print(attn_weights.shape)  # Expected: (2, 4, 10, 10)
    """

    __constants__ = ["mask"]
    mask: bool

    def __init__(self, mask=False):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.mask = mask

    def __repr__(self):
        return f"{self.__class__.__name__}(mask={self.mask})"

    def forward(self, qkv):
        r"""
        Args:
            qkv (tuple of Tensor): Tuple of (Q, K, V), each with shape
                (batch_size, seq_len, num_heads, dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, num_heads, dim)
            Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        q, k, v = qkv
        d_k = k.size(-1)

        # Transpose to get shape (batch_size, num_heads, seq_len, dim)
        q, k, v = q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2)

        # Compute scaled attention scores
        scores = torch.matmul(q, k.mT) / (d_k**0.5)

        # Apply causal mask if enabled (assuming q and k have same seq_len)
        if self.mask:
            seq_len = q.size(-2)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            scores = scores + mask * float("-inf")

        # Apply softmax and attend to values, reshape back to original shape
        attn = self.softmax(scores)
        z = torch.matmul(attn, v).transpose(-3, -2)

        return z, attn
