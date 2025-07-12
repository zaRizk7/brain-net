import torch
from torch import nn

__all__ = ["EdgeToEdgeConv", "EdgeToNodeConv", "NodeToGraphConv"]


class BaseConv(nn.Module):
    r"""
    Base class for specialized convolutional layers operating on structured tensors.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        spatial_size (int): Size of the spatial dimension (e.g., number of nodes/ROIs).
        bias (bool): Whether to include a bias term. Default: True.
        device (torch.device or str, optional): Device on which to allocate parameters.
        dtype (torch.dtype, optional): Desired data type of the parameters.
    """

    def __init__(self, in_channels, out_channels, spatial_size, bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size

        self.init_parameters(bias, **factory_kwargs)
        self.reset_parameters()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, spatial_size={self.spatial_size}, "
            f"bias={self.bias is not None})"
        )

    def init_weight(self, **kwargs):
        """
        Initialize the weight tensor with shape:
        (out_channels, in_channels, spatial_size).
        """
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.spatial_size, **kwargs))

    def init_bias(self, bias, **kwargs):
        """
        Initialize the bias tensor with shape (out_channels, 1) if bias is True.
        """
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, 1, **kwargs))
        else:
            self.register_parameter("bias", None)

    def init_parameters(self, bias, **kwargs):
        """
        Initialize weight and bias parameters.
        """
        self.init_weight(**kwargs)
        self.init_bias(bias, **kwargs)

    def reset_parameters(self):
        """
        Reset the weight and bias using Xavier initialization and zero-initialization respectively.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class EdgeToEdgeConv(BaseConv):
    r"""
    Edge-to-Edge convolution layer. Projects each edge in the input to another edge space
    by applying row-wise and column-wise projections independently and summing the result.

    The weight and bias are split into separate components for row and column transformations.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        spatial_size (int): Size of the spatial dimension (e.g., number of nodes/ROIs).
        bias (bool): Whether to include a bias term. Default: True.
        device (torch.device or str, optional): Device on which to allocate parameters.
        dtype (torch.dtype, optional): Desired data type of the parameters.
    """

    def init_weight(self, **kwargs):
        self.weight_row = nn.Parameter(torch.empty(self.out_channels, self.in_channels, self.spatial_size, **kwargs))
        self.weight_col = nn.Parameter(self.weight_row.data.clone())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, spatial_size={self.spatial_size}, "
            f"bias={self.bias_row is not None})"
        )

    def init_bias(self, bias, **kwargs):
        if bias:
            self.bias_row = nn.Parameter(torch.empty(self.out_channels, 1, **kwargs))
            self.bias_col = nn.Parameter(self.bias_row.data.clone())
        else:
            self.register_parameter("bias_row", None)
            self.register_parameter("bias_col", None)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_row)
        nn.init.xavier_uniform_(self.weight_col)
        if self.bias_row is not None:
            nn.init.zeros_(self.bias_row)
            nn.init.zeros_(self.bias_col)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (..., in_channels, spatial_size, spatial_size)

        Returns:
            Tensor: Output tensor of shape (..., out_channels, spatial_size, spatial_size)
        """
        bias_row = self.bias_row if self.bias_row is not None else 0
        bias_col = self.bias_col if self.bias_col is not None else 0

        z_row = torch.einsum("...cij,dcj->...di", x, self.weight_row) + bias_row
        z_col = torch.einsum("...cij,dci->...dj", x, self.weight_col) + bias_col

        return z_row[..., None, :] + z_col[..., :, None]


class EdgeToNodeConv(BaseConv):
    r"""
    Edge-to-Node convolution layer. Aggregates edge information across rows
    to produce node-level features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        spatial_size (int): Size of the spatial dimension (e.g., number of nodes/ROIs).
        bias (bool): Whether to include a bias term. Default: True.
        device (torch.device or str, optional): Device on which to allocate parameters.
        dtype (torch.dtype, optional): Desired data type of the parameters.
    """

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (..., in_channels, spatial_size, spatial_size)

        Returns:
            Tensor: Output tensor of shape (..., out_channels, spatial_size)
        """
        bias = self.bias if self.bias is not None else 0
        return torch.einsum("...cij,dcj->...di", x, self.weight) + bias


class NodeToGraphConv(BaseConv):
    r"""
    Node-to-Graph convolution layer. Aggregates node-level features into graph-level
    representations by collapsing over the spatial dimension.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        spatial_size (int): Size of the spatial dimension (e.g., number of nodes/ROIs).
        bias (bool): Whether to include a bias term. Default: True.
        device (torch.device or str, optional): Device on which to allocate parameters.
        dtype (torch.dtype, optional): Desired data type of the parameters.
    """

    def init_bias(self, bias, **kwargs):
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, **kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (..., in_channels, spatial_size)

        Returns:
            Tensor: Output tensor of shape (..., out_channels)
        """
        bias = self.bias if self.bias is not None else 0
        return torch.einsum("...ci,dci->...d", x, self.weight) + bias
