import torch
from torch import nn

__all__ = ["Residual"]


class Residual(nn.Module):
    """
    Residual module that applies a given submodule to an input tensor and combines the submodule's output
    with the original input using a specified interaction method. This design allows for flexible
    skip/residual connections, which can help with gradient flow and model expressivity.

    The submodule is applied to the input, and its output is optionally passed through a dropout layer
    (if `p_dropout` is specified). The result is then combined with the original input using one of several
    supported interaction types:
        - 'add': Element-wise addition of the input and submodule output (standard residual connection).
        - 'mul': Element-wise multiplication of the input and submodule output.
        - 'concat': Concatenation of the input and submodule output along the last dimension.

    If the submodule returns a tuple (e.g., for modules that return auxiliary outputs), only the first element
    of the tuple is combined with the input, and any additional elements are returned unchanged.

    Args:
        module (nn.Module): The submodule to apply to the input.
        p_dropout (float, optional): Dropout probability to apply after the submodule. If `None`, no dropout is applied.
        interaction (str): The type of interaction to combine input and submodule output.
            Supported types are 'add', 'concat', and 'mul'. Default: 'add'.

    Example:
        >>> res = Residual(nn.Linear(10, 10), p_dropout=0.1, interaction='add')
        >>> x = torch.randn(2, 10)
        >>> y = res(x)
    """

    def __init__(self, module, p_dropout=None, interaction="add"):
        # Validate interaction type
        if interaction not in {"add", "concat", "mul"}:
            raise ValueError(
                f"Unsupported interaction type: {interaction}. Supported types are 'add', 'concat', 'mul'."
            )
        super().__init__()
        # Store the submodule to be applied to the input
        self.module = module
        # Optionally apply dropout after the submodule; otherwise, use identity
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else nn.Identity()
        # Store the interaction type for combining input and output
        self.interaction = interaction

    def __repr__(self):
        # Custom string representation to show module, dropout, and interaction type
        p_dropout = self.dropout.p if isinstance(self.dropout, nn.Dropout) else None
        return (
            f"{self.__class__.__name__}(\n"
            f"   module={self.module},\n"
            f"   p_dropout={p_dropout},\n"
            f"   interaction='{self.interaction}'\n"
            f")"
        )

    def forward(self, x, *args, **kwargs):
        """
        Forward pass that applies the submodule to the input and combines the result with the input.

        If the submodule returns a tuple, only the first element is combined with the input;
        the remaining elements are returned as-is.
        """
        # Apply the submodule to the input (and any additional kwargs)
        z = self.module(x, *args, **kwargs)
        if not isinstance(z, tuple):
            # If the output is a tensor, apply interaction and return
            return self._apply_interaction(x, z)
        # If the output is a tuple, apply interaction to the first element
        first = self._apply_interaction(x, z[0])
        if len(z) == 1:
            # Only one element in tuple, return as single-element tuple
            return (first,)
        # Return a tuple: (combined first element, rest of original tuple)
        return (first,) + z[1:]

    def _apply_interaction(self, x, z):
        # Apply dropout to the submodule output (if enabled)
        z = self.dropout(z)
        # Combine input and output according to the specified interaction type
        if self.interaction == "add":
            return x + z
        elif self.interaction == "mul":
            return x * z
        # For 'concat', concatenate along the last dimension
        return torch.cat([x, z], dim=-1)
