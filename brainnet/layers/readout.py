import torch
from torch import nn

__all__ = ["OCRead"]


class OCRead(nn.Module):
    r"""
    Orthonormal Clustering Readout (OCRead) module.

    OCRead learns a set of orthonormal cluster centers and uses them to extract
    meaningful summaries of input features via soft assignment. Given an input
    tensor `x`, it computes similarity to the learned orthonormal centers,
    applies a softmax over the similarities to obtain assignment weights, and
    returns a per-cluster weighted sum of input features.

    This operation is useful for interpretable representation learning across
    multiple regions (e.g., ROIs) in a batch-wise manner.

    Args:
        num_clusters (int): Number of clusters (centers) to learn.
        num_embeddings (int): Dimension of the input embeddings (e.g., number of ROIs).
        ortho_init (bool): If True, use Gram-Schmidt orthonormalization when initializing
            the cluster centers to ensure they are orthonormal. Default: True.
        learnable (bool): If True, the cluster centers are learnable parameters. Default: True.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.

    Shape:
        - Input: `(batch_size, num_rois, num_embeddings)`
        - Output: `(batch_size, num_clusters, num_embeddings)`

    Variables:
        - `centers`: Learnable orthonormal cluster centers of shape `(num_clusters, num_embeddings)`.

    Examples:
        >>> ocread = OCRead(num_clusters=10, num_embeddings=200)
        >>> x = torch.randn(8, 200, 200)  # Batch of 8, 200 ROIs, 200 embeddings
        >>> out = ocread(x)
        >>> print(out.shape)  # Expected output shape: (8, 10, 200)
    """

    __constants__ = ["num_clusters", "num_embeddings", "ortho_init"]
    num_clusters: int
    num_embeddings: int
    ortho_init: bool
    centers: torch.Tensor
    softmax: nn.Softmax

    def __init__(self, num_clusters, num_embeddings, ortho_init=True, learnable=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_clusters = num_clusters
        self.num_embeddings = num_embeddings
        self.ortho_init = ortho_init

        # Learnable orthonormal cluster centers (num_clusters x num_embeddings)
        self.centers = nn.Parameter(torch.empty(num_clusters, num_embeddings, **factory_kwargs), learnable)

        # Softmax over similarity scores
        self.softmax = nn.Softmax(dim=-1)

        self.reset_parameters()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_clusters={self.num_clusters}, "
            f"num_embeddings={self.num_embeddings}, "
            f"ortho_init={self.ortho_init}, "
            f"learnable={self.centers.requires_grad})"
        )

    def reset_parameters(self):
        r"""
        Initializes cluster centers with orthonormality constraint.

        The initialization process is:
        1. Xavier uniform initialization for diverse directionality.
        2. Gram-Schmidt orthonormalization to enforce center orthogonality.

        This mirrors the initialization strategy described in the original
        Orthonormal Clustering Readout (OCRead) method.
        """
        nn.init.xavier_uniform_(self.centers)

        if not self.ortho_init:
            self.centers.data /= torch.norm(self.centers.data, dim=-1, keepdim=True)
            return

        # Gram-Schmidt orthonormalization
        for k in range(self.num_clusters):
            c_k = self.centers[k].data.clone()
            for j in range(k):
                u_j = self.centers.data[j].clone()
                c_k -= torch.dot(c_k, u_j) / torch.dot(u_j, u_j) * u_j
            self.centers.data[k] = c_k / torch.norm(c_k)

    def forward(self, x, return_assignments=False):
        r"""
        Forward pass for OCRead.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_rois, num_embeddings)
            return_assignments (bool): If True, also return the assignment weights.
                Default: False.

        Returns:
            Tensor: Clustered readout tensor of shape (batch_size, num_clusters, num_embeddings),
                    computed as the aggreegated embeddings per cluster.
            Tensor (optional): Assignment weights of shape (batch_size, num_rois, num_clusters),
                    representing the soft assignment of each ROI to each cluster.
        """
        # Compute assignment weights via dot-product similarity
        # x: (batch_size, num_rois, num_embeddings)
        # Transpose centers: (num_embeddings, num_clusters)
        # p: (batch_size, num_rois, num_clusters)
        p = self.softmax(torch.matmul(x, self.centers.mT))

        # Weighted sum over input vectors per cluster
        # Transpose p: (batch_size, num_clusters, num_rois)
        # x:           (batch_size, num_rois, num_embeddings)
        # Output:      (batch_size, num_clusters, num_embeddings)
        z = torch.matmul(p.mT, x)

        if not return_assignments:
            return z
        return z, p
