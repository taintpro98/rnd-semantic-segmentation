import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, eps=1):
        super().__init__()

        self.eps = eps
        self.alpha = alpha

    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        probs = torch.sigmoid(pred)
        true_pos = torch.sum(probs * label, dim=[0, 2, 3])
        false_neg = torch.sum(label * (1 - probs), dim=[0, 2, 3])
        false_pos = torch.sum(probs * (1 - label), dim=[0, 2, 3])
        return 1 - torch.mean(
            (true_pos + self.eps) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.eps)
        )

class MultiscaleLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, predicts, labels):
        # loss = torch.scalar_tensor(0, device=self._device)
        loss = torch.scalar_tensor(0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        for pred, label in zip(predicts, labels):
            loss += self.loss_fn(pred, label)
        return loss

class CompoundLoss(nn.Module):
    def __init__(self, losses: Iterable, weights=None):
        super().__init__()

        if weights is None:
            N = len(losses)
            weights = [1./N] * N

        self.weights = weights
        self.losses = nn.ModuleList(losses)

    def forward(self, *inputs):
        """
        Forward function.
        Sums all losses over the given inputs
        """
        # loss = torch.scalar_tensor(0, device=self._device)
        loss = torch.scalar_tensor(0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        for loss_fn, w in zip(self.losses, self.weights):
            loss += w * loss_fn(*inputs)

        return loss

class BinaryCrossEntropyLoss(nn.Module):
    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        return F.binary_cross_entropy_with_logits(pred, label.float())