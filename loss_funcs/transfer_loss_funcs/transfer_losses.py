import torch
from torch import nn

from loss_funcs.transfer_loss_funcs.RDMAlignLoss import RDMAlignLoss
from loss_funcs.transfer_loss_funcs.adv import AdversarialLoss
from loss_funcs.transfer_loss_funcs.coral import CORAL
from loss_funcs.transfer_loss_funcs.daan import DAANLoss
from loss_funcs.transfer_loss_funcs.mmd import MMDLoss


class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super(TransferLoss, self).__init__()
        self.loss_type = loss_type
        # lmmd seems to have some problems, so it is not included
        if loss_type == "dann":
            self.loss_func = AdversarialLoss(**kwargs)
        elif loss_type == "mmd":
            self.loss_func = MMDLoss(**kwargs)
        elif loss_type == "coral":
            self.loss_func = CORAL
        elif loss_type == "daan":
            self.loss_func = DAANLoss(**kwargs)
        elif loss_type == "rdm":
            self.loss_func = RDMAlignLoss(distance='euclidean')  # or 'cosine'
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y: torch.tensor(0.0, device=x.device)

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)

    def get_parameters(self):
        params = []
        if self.loss_type == "dann":
            params.append(
                {"params": self.loss_func.domain_classifier.parameters(), "lr_mult": 1}
            )
        elif self.loss_type == "daan":
            params.append(
                {'params': self.loss_func.domain_classifier.parameters(), "lr_mult": 1}
            )
            params.append(
                {'params': self.loss_func.local_classifiers.parameters(), "lr_mult": 1}
            )
        return params
