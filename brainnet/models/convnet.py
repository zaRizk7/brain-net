from torch import nn

from ..layers import EdgeToEdgeConv, EdgeToNodeConv, NodeToGraphConv

__all__ = ["BrainNetCNN"]


class BrainNetCNN(nn.Sequential):
    """
    BrainNetCNN: A convolutional neural network architecture designed for processing
    brain connectivity matrices (e.g., functional or structural connectomes).

    The network consists of a sequence of specialized convolutional layers:
        - Edge-to-Edge Convolution (E2E): Learns relationships between edges in the connectivity matrix.
        - Edge-to-Node Convolution (E2N): Aggregates edge features into node-level representations.
        - Node-to-Graph Convolution (N2G): Aggregates node features into a graph-level embedding.
        - (Optional) Final Linear Layer: Maps graph-level embeddings to target predictions (e.g., class scores).

    This architecture follows the design principles from the original BrainNetCNN paper.

    Args:
        channels (List[int]): A list of channel dimensions for each convolutional stage.
            Example: [1, 32, 64, 128, 256] would imply:
                - E2E layers: 1→32, 32→64
                - E2N: 64→128
                - N2G: 128→256
        spatial_size (int): The size (number of ROIs) of the input connectivity matrix (assumed square).
        num_outputs (int, optional): If specified, adds a final linear layer with this output size. Default: `None`.
        bias (bool): Whether to include biases in the convolutional and linear layers. Default: `True`.
        negative_slope (float): Negative slope for LeakyReLU activations. Default: 0.33.
        device (torch.device, optional): Device to initialize model parameters on.
        dtype (torch.dtype, optional): Data type for model parameters.

    Examples:
        >>> model = BrainNetCNN([1, 32, 64, 128, 256], spatial_size=200, num_outputs=2)
        >>> x = torch.randn(8, 1, 200, 200)  # Batch of 8, 1 channel, 200x200 connectivity matrices
        >>> out = model(x)
        >>> print(out.shape)  # Expected output shape: (8, 2) if num_outputs=2
    """

    def __init__(
        self, channels, spatial_size, num_outputs=None, bias=True, negative_slope=0.33, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Add Edge-to-Edge convolutional layers (multiple layers if len(channels) > 4)
        for i in range(1, len(channels) - 2):
            in_channels = channels[i - 1]
            out_channels = channels[i]
            self.add_module(
                f"e2e_{i:0=2d}", EdgeToEdgeConv(in_channels, out_channels, spatial_size, bias, **factory_kwargs)
            )
            self.add_module(f"relu_{i:0=2d}", nn.LeakyReLU(negative_slope))

        # Edge-to-Node convolution
        in_channels = channels[-3]
        out_channels = channels[-2]
        self.add_module("e2n", EdgeToNodeConv(in_channels, out_channels, spatial_size, bias, **factory_kwargs))
        self.add_module(f"relu_{i+1:0=2d}", nn.LeakyReLU(negative_slope))

        # Node-to-Graph convolution
        in_channels = channels[-2]
        out_channels = channels[-1]
        self.add_module("n2g", NodeToGraphConv(in_channels, out_channels, spatial_size, bias, **factory_kwargs))

        # Optional final linear layer for classification or regression
        if num_outputs is not None:
            self.add_module("linear", nn.Linear(out_channels, num_outputs, **factory_kwargs))
