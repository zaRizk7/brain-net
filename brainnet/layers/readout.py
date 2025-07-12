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
        num_features (int): Dimensionality of input features.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.

    Shape:
        - Input: `(batch_size, num_rois, num_features)`
        - Output: `(batch_size, num_clusters, num_features)`
    """

    def __init__(self, num_clusters, num_features, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features

        # Learnable orthonormal cluster centers (num_clusters x num_features)
        self.centers = nn.Parameter(torch.empty(num_clusters, num_features, **factory_kwargs))

        # Softmax over similarity scores
        self.softmax = nn.Softmax(dim=-1)

        self.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}(num_clusters={self.num_clusters}, num_features={self.num_features})"

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

        # Gram-Schmidt orthonormalization
        for k in range(self.num_clusters):
            c_k = self.centers[k].data.clone()
            for j in range(k):
                u_j = self.centers.data[j].clone()
                c_k -= torch.dot(c_k, u_j) / torch.dot(u_j, u_j) * u_j
            self.centers.data[k] = c_k / torch.norm(c_k)

    def forward(self, x):
        r"""
        Forward pass for OCRead.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_rois, num_features)

        Returns:
            Tensor: Clustered readout tensor of shape (batch_size, num_clusters, num_features),
                    computed as a soft assignment-weighted sum over input ROIs.
        """
        # Compute assignment weights via dot-product similarity
        # Shape: (batch_size, num_rois, num_clusters)
        p = self.softmax(torch.matmul(x, self.centers.mT))

        # Weighted sum over input vectors per cluster
        # Transpose p: (batch_size, num_clusters, num_rois)
        # x:           (batch_size, num_rois, num_features)
        # Output:      (batch_size, num_clusters, num_features)
        return torch.matmul(p.mT, x)
