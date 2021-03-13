import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from loguru import logger
from typing import Optional, Callable, Union, Dict, IO

from polypnet.utils import generate_scales, probs_to_onehot, CosineAnnealingWarmupLR
from polypnet.losses import MultiscaleLoss, CompoundLoss,\
    DiceLoss, TverskyLoss, BinaryCrossEntropyLoss
from polypnet import metrics


class UnetModelWrapper(pl.LightningModule):
    def __init__(self,
        model: nn.Module,
        optimizer: nn.Module,
        loss_fn: Optional[Callable] = None,
        num_classes=1
    ):
        super().__init__()

        if loss_fn is None:
            loss_fn = MultiscaleLoss(
                CompoundLoss([
                    TverskyLoss(),
                    BinaryCrossEntropyLoss()
                ])
            )

        self.optimizer = optimizer
        self.model = model
        self.loss_fn = loss_fn
        self.num_classes = num_classes

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [CosineAnnealingWarmupLR(self.optimizer, T_max=50, warmup_epochs=5)]
        )

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        input = batch[0]  # B x C x H x W
        label = batch[1]  # B x C x H x W
        if self.num_classes > 1:
            label = F.one_hot(label.squeeze(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        scaled_labels = generate_scales(label, self.model.output_scales)

        outputs = self(input)  # [B x C x H x W]

        output = outputs[0]
        pred_probs = torch.sigmoid(output)  # B x C x H x W
        pred_probs = probs_to_onehot(pred_probs)
        loss = self.loss_fn(outputs, scaled_labels)

        # Calculate metrics
        self._log_metrics(pred_probs, label, loss, prefix="train")

        return loss

    def validation_step(self, batch, batch_idx):
        input = batch[0]  # B x C x H x W
        label = batch[1]  # B x C x H x W
        if self.num_classes > 1:
            label = F.one_hot(label.squeeze(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        scaled_labels = generate_scales(label, self.model.output_scales)

        outputs = self(input)  # [B x C x H x W]

        output = outputs[0]
        pred_probs = torch.sigmoid(output)  # B x C x H x W
        pred_probs = probs_to_onehot(pred_probs)
        loss = self.loss_fn(outputs, scaled_labels)

        # Calculate metrics
        self._log_metrics(pred_probs, label, loss, prefix="val")

    def test_step(self, batch, batch_idx):
        input = batch[0]  # B x C x H x W
        label = batch[1]  # B x C x H x W

        batch_size = input.shape[0]
        if self.num_classes > 1:
            label = F.one_hot(label.squeeze(dim=1), self.num_classes).permute(0, 3, 1, 2).float()

        outputs = self(input)  # [B x C x H x W]
        output = outputs[0]
        pred_probs = torch.sigmoid(output)  # B x C x H x W
        pred_probs = probs_to_onehot(pred_probs)

        total_iou = metrics.iou(pred_probs, label) * batch_size
        total_dice = metrics.dice(pred_probs, label) * batch_size
        total_precision = metrics.precision(pred_probs, label) * batch_size
        total_recall = metrics.recall(pred_probs, label) * batch_size

        total_intersection = torch.sum(pred_probs * label, dim=[0, 2, 3])
        total_union = torch.sum(pred_probs, dim=[0, 2, 3]) + torch.sum(label, dim=[0, 2, 3])
        total_true_pos = torch.sum(pred_probs * label, dim=[0, 2, 3])
        total_all_pos = torch.sum(pred_probs == 1, dim=[0, 2, 3])
        total_all_true = torch.sum(label == 1, dim=[0, 2, 3])

        return {
            "total_intersection": total_intersection,
            "total_union": total_union,
            "total_true_pos": total_true_pos,
            "total_all_pos": total_all_pos,
            "total_all_true": total_all_true,
            "total_iou": total_iou,
            "total_dice": total_dice,
            "total_precision": total_precision,
            "total_recall": total_recall,
            "batch_size": batch_size
        }

    def test_epoch_end(self, outputs):
        sum_batch_size = sum([
            o["batch_size"]
            for o in outputs
        ])
        sum_iou = sum([
            o["total_iou"]
            for o in outputs
        ])
        sum_dice = sum([
            o["total_dice"]
            for o in outputs
        ])
        sum_precision = sum([
            o["total_precision"]
            for o in outputs
        ])
        sum_recall = sum([
            o["total_recall"]
            for o in outputs
        ])
        sum_intersection = sum([
            o["total_intersection"]
            for o in outputs
        ])
        sum_union = sum([
            o["total_union"]
            for o in outputs
        ])
        sum_true_pos = sum([
            o["total_true_pos"]
            for o in outputs
        ])
        sum_all_pos = sum([
            o["total_all_pos"]
            for o in outputs
        ])
        sum_all_true = sum([
            o["total_all_true"]
            for o in outputs
        ])

        iou = sum_iou / sum_batch_size
        dice = sum_dice / sum_batch_size
        precision = sum_precision / sum_batch_size
        recall = sum_recall / sum_batch_size

        self.log("test.iou.macro", iou, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test.dice.macro", dice, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test.precision.macro", precision, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test.recall.macro", recall, prog_bar=False, on_step=False, on_epoch=True)

        micro_iou = torch.mean((sum_intersection + 1) / (sum_union - sum_intersection + 1))
        micro_dice = torch.mean((2 * sum_intersection + 1) / (sum_union + 1))
        micro_precision = torch.mean((sum_true_pos + 1) / (sum_all_pos + 1))
        micro_recall = torch.mean((sum_true_pos + 1) / (sum_all_true + 1))

        self.log("test.iou.micro", micro_iou, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test.dice.micro", micro_dice, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test.precision.micro", micro_precision, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test.recall.micro", micro_recall, prog_bar=False, on_step=False, on_epoch=True)

    def _log_metrics(self, pred_probs, label, loss, prefix: str):
        iou = metrics.iou(pred_probs, label)
        self.log(f"{prefix}.iou", iou, prog_bar=False, on_step=False, on_epoch=True)

        dice = metrics.dice(pred_probs, label)
        self.log(f"{prefix}.dice", dice, prog_bar=False, on_step=False, on_epoch=True)

        precision = metrics.precision(pred_probs, label)
        self.log(f"{prefix}.precision", precision, prog_bar=False, on_step=False, on_epoch=True)

        recall = metrics.recall(pred_probs, label)
        self.log(f"{prefix}.recall", recall, prog_bar=False, on_step=False, on_epoch=True)

        self.log(f"{prefix}.loss", loss, on_step=False, on_epoch=True)
