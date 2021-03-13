import torch 

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