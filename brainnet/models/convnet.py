import torch
from torch import nn

from ..layers import EdgeToEdgeConv, EdgeToNodeConv, NodeToGraphConv


class BrainNetCNN(nn.Sequential):

    def __init__(
        self, channels, spatial_size, num_outputs=None, bias=True, negative_slope=0.33, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Assume the last two layers are EdgeToNodeConv and NodeToGraphConv
        for i in range(1, len(channels) - 2):
            in_channels = channels[i - 1]
            out_channels = channels[i]
            self.add_module(
                f"e2e_{i:0=2d}", EdgeToEdgeConv(in_channels, out_channels, spatial_size, bias, **factory_kwargs)
            )
            self.add_module(f"relu_{i:0=2d}", nn.LeakyReLU(negative_slope))

        # Edge to Node convolution
        in_channels = channels[-3]
        out_channels = channels[-2]
        self.add_module("e2n", EdgeToNodeConv(in_channels, out_channels, spatial_size, bias, **factory_kwargs))
        self.add_module(f"relu_{i+1:0=2d}", nn.LeakyReLU(negative_slope))

        # Node to Graph convolution
        in_channels = channels[-2]
        out_channels = channels[-1]
        self.add_module("n2g", NodeToGraphConv(in_channels, out_channels, spatial_size, bias, **factory_kwargs))

        if num_outputs is not None:
            # Final linear layer for classification/regression
            self.add_module("linear", nn.Linear(out_channels, num_outputs, **factory_kwargs))
