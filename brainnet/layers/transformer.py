from torch import nn

from .multi_head import MultiHeadAttention
from .residual import Residual

__all__ = ["TFEncoder", "TFDecoder"]


class TFEncoder(nn.Module):
    """
    A Transformer Encoder block composed of multi-head attention, feedforward network,
    layer normalization, and optional dropout. This module can be used as a building
    block for deep transformer architectures in sequence modeling tasks.

    Args:
        num_inputs (int): Number of input features.
        num_heads (int): Number of attention heads. Default: 1.
        num_keys (int, optional): Number of key features. If None, defaults to num_inputs.
        num_values (int, optional): Number of value features. If None, defaults to num_inputs.
        num_feedforward (int, optional): Dimensionality of the feedforward layer. If None, defaults to 4 * num_inputs.
        num_outputs (int, optional): Number of output features. If None, defaults to num_inputs.
        p_dropout (float, optional): Dropout probability. If None, dropout is not applied.
        bias (bool): Whether to use bias terms in linear layers and normalization. Default: True.
        norm_affine (bool): Whether to use affine transformation in LayerNorm. Default: True.
        activation (nn.Module, optional): Activation function for the feedforward network. Default: nn.GELU().
        device (torch.device, optional): Device for module parameters.
        dtype (torch.dtype, optional): Data type for module parameters.
    """

    def __init__(
        self,
        num_inputs,
        num_heads=1,
        num_keys=None,
        num_values=None,
        num_feedforward=None,
        num_outputs=None,
        p_dropout=None,
        bias=True,
        norm_affine=True,
        activation=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        num_outputs = num_outputs or num_inputs
        num_feedforward = num_feedforward or num_inputs * 4
        activation = activation or nn.GELU()
        super().__init__()
        self.mha = Residual(
            MultiHeadAttention(num_inputs, num_heads, num_keys, num_values, num_outputs, bias, **factory_kwargs),
            p_dropout,
        )
        self.norm_mha = nn.LayerNorm(num_outputs, elementwise_affine=norm_affine, bias=bias, **factory_kwargs)

        self.ffn = Residual(
            nn.Sequential(
                nn.Linear(num_outputs, num_feedforward, **factory_kwargs),
                activation,
                nn.Linear(num_feedforward, num_outputs, **factory_kwargs),
            ),
            p_dropout,
        )
        self.norm_ffn = nn.LayerNorm(num_outputs, elementwise_affine=norm_affine, bias=bias, **factory_kwargs)

    def forward(self, x, z=None, return_attention=False, mask=None, causal_mask=False):
        """
        Forward pass for the Transformer encoder block.

        Args:
            x (Tensor): Input tensor of shape (..., seq_len, num_inputs), where seq_len is the sequence length.
            z (Tensor, optional): Optional secondary input for cross-attention, of shape (..., seq_len, num_inputs).
                If provided, it is used as the key and value for the attention mechanism; otherwise, self-attention is performed.
            return_attention (bool): If True, also returns the attention weights from the multi-head attention layer.
            mask (Tensor, optional): Optional mask tensor to apply to attention scores.
                If `None`, no mask is applied. Default: `None`.
            causal_mask (bool): If `True`, applies a causal mask to prevent attending to future positions.
                Overrides the `mask` argument if set to `True`. Default: `False`.

        Returns:
            Tensor: The output tensor after attention, normalization, and feedforward processing,
                of shape (..., seq_len, num_outputs).
            Tensor (optional): If `return_attention` is True, returns a tuple where the second element
                is the attention weights tensor of shape (..., num_heads, seq_len, seq_len).
        """
        x = self.mha(x, z, return_attention, mask, causal_mask)
        if return_attention:
            x, attn = x
        x = self.norm_mha(x)
        x = self.ffn(x)
        x = self.norm_ffn(x)

        return (x, attn) if return_attention else x


class TFDecoder(nn.Module):
    """
    A Transformer Encoder block composed of multi-head attention, feedforward network,
    layer normalization, and optional dropout. This module can be used as a building
    block for deep transformer architectures in sequence modeling tasks.

    Args:
        num_inputs (int): Number of input features.
        num_heads (int): Number of attention heads. Default: 1.
        num_keys (int, optional): Number of key features. If None, defaults to num_inputs.
        num_values (int, optional): Number of value features. If None, defaults to num_inputs.
        num_feedforward (int, optional): Dimensionality of the feedforward layer. If None, defaults to 4 * num_inputs.
        num_outputs (int, optional): Number of output features. If None, defaults to num_inputs.
        p_dropout (float, optional): Dropout probability. If None, dropout is not applied.
        bias (bool): Whether to use bias terms in linear layers and normalization. Default: True.
        norm_affine (bool): Whether to use affine transformation in LayerNorm. Default: True.
        activation (nn.Module, optional): Activation function for the feedforward network. Default: nn.GELU().
        device (torch.device, optional): Device for module parameters.
        dtype (torch.dtype, optional): Data type for module parameters.
    """

    def __init__(
        self,
        num_inputs,
        num_heads=1,
        num_keys=None,
        num_values=None,
        num_feedforward=None,
        num_outputs=None,
        p_dropout=None,
        bias=True,
        norm_affine=True,
        activation=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        num_outputs = num_outputs or num_inputs
        num_feedforward = num_feedforward or num_inputs * 4
        activation = activation or nn.GELU()
        super().__init__()
        self.mha = Residual(
            MultiHeadAttention(num_inputs, num_heads, num_keys, num_values, num_outputs, bias, **factory_kwargs),
            p_dropout,
        )

        self.norm_mha = nn.LayerNorm(num_outputs, elementwise_affine=norm_affine, bias=bias, **factory_kwargs)

        # Encoder can be re-used for the second part of the decoder
        self.encoder = TFEncoder(
            num_inputs,
            num_heads,
            num_keys,
            num_values,
            num_feedforward,
            num_outputs,
            p_dropout,
            bias,
            norm_affine,
            activation,
            **factory_kwargs,
        )

    def forward(
        self, x, z=None, return_attention=False, s_mask=None, c_mask=None, s_causal_mask=False, c_causal_mask=False
    ):
        """
        Forward pass for the Transformer decoder block.

        Args:
            x (Tensor): Input tensor of shape (..., seq_len, num_inputs), where seq_len is the sequence length.
            z (Tensor, optional): Optional secondary input for cross-attention, of shape (..., seq_len, num_inputs).
                If provided, it is used as the key and value for the attention mechanism; otherwise, self-attention is performed.
            return_attention (bool): If True, also returns the attention weights from the multi-head attention layer.
            s_mask (Tensor, optional): Optional mask tensor for self-attention.
                If `None`, no mask is applied. Default: `None`.
            c_mask (Tensor, optional): Optional mask tensor for cross-attention.
                If `None`, no mask is applied. Default: `None`.
            s_causal_mask (bool): If `True`, applies a causal mask to self-attention to prevent attending to future positions.
                Overrides `s_mask` if set to `True`.
            c_causal_mask (bool): If `True`, applies a causal mask to cross-attention to prevent attending to future positions.
                Overrides `c_mask` if set to `True`.

        Returns:
            Tensor: The output tensor after attention, normalization, and feedforward processing,
                of shape (..., seq_len, num_outputs).
            Tensor (optional): If `return_attention` is True, returns a tuple where the second element
                is the self-attention weights tensor of shape (..., num_heads, seq_len, seq_len).
            Tensor (optional): If `return_attention` is True, returns a tuple where the third element
                is the cross-attention weights tensor of shape (..., num_heads, seq_len, seq_len).
        """
        x = self.mha(x, x, return_attention, s_mask, s_causal_mask)
        if return_attention:
            x, s_attn = x
        x = self.norm_mha(x)
        x = self.encoder(x, z, return_attention, c_mask, c_causal_mask)
        if return_attention:
            x, c_attn = x
        return (x, s_attn, c_attn) if return_attention else x
