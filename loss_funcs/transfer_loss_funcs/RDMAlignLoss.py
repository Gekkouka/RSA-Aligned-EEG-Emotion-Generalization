import torch
import torch.nn as nn
import torch.nn.functional as F

class RDMAlignLoss(nn.Module):
    def __init__(self, distance='euclidean', reduction='mean'):
        super(RDMAlignLoss, self).__init__()
        assert distance in ['euclidean', 'cosine'], "Only euclidean and cosine supported."
        self.distance = distance
        self.reduction = reduction

    def compute_rdm(self, X):
        # X: [batch_size, feat_dim]
        if self.distance == 'euclidean':
            # Compute squared Euclidean distance matrix
            xx = (X ** 2).sum(dim=1, keepdim=True)  # [N,1]
            dist = xx + xx.t() - 2 * (X @ X.t())    # [N,N]
            dist = torch.clamp(dist, min=1e-6)      # Numerical stability
        elif self.distance == 'cosine':
            # Normalize first
            X_norm = F.normalize(X, p=2, dim=1)
            dist = 1 - X_norm @ X_norm.t()
        return dist

    def forward(self, source, target):
        # source/target: [batch_size, feat_dim] — one-to-one matched
        rdm_s = self.compute_rdm(source)
        rdm_t = self.compute_rdm(target)

        # Vectorize upper triangle without diagonal
        def upper_tri(mat):
            idx = torch.triu_indices(mat.size(0), mat.size(1), offset=1)
            return mat[idx[0], idx[1]]

        rdm_s_vec = upper_tri(rdm_s)
        rdm_t_vec = upper_tri(rdm_t)

        loss = F.mse_loss(rdm_s_vec, rdm_t_vec, reduction=self.reduction)
        return loss
