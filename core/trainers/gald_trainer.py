import os

import torch
import torch.nn.functional as F

from base.base_trainer import BaseTrainer
from core.models.classifiers.gcpacc.gcpa_cc2 import GCPADecoder, GCPAEncoder
from core.utils.adapt_lr import CosineAnnealingWarmupLR, GradualWarmupScheduler

def expand_target(x, n_class, mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

def GeneralizedDiceLoss(output, target, eps=1e-5, weight_type='square'):  # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()

    if target.dim() == 4:
        # target[target == 4] = 3  # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

class GALDTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank):
        super(GALDTrainer, self).__init__(name, cfg, train_loader, local_rank, logger)

    def init_params(self):
        self.encoder = GCPAEncoder()
        self.decoder = GCPADecoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.optimizer_enc = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.SOLVER.BASE_LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=self.cfg.SOLVER.BASE_LR*10, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

    def _save_checkpoint(self, epoch, save_path):
        checkpoint = {
            'epoch': epoch, 
            'iteration': self.iteration, 
            'encoder': self.encoder.state_dict(), 
            'decoder': self.decoder.state_dict(),
            'optimizer_enc': self.optimizer_fea.state_dict(), 
            'optimizer_dec': self.optimizer_cls.state_dict()
        }
        torch.save(checkpoint, save_path)

    def _train_epoch(self, epoch):
        for i, (src_input, src_label) in enumerate(self.train_loader):
            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()

            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            hardnetout = self.encoder(src_input)
            out5, out4, out3, out2 = self.decoder(hardnetout)

            loss5 = GeneralizedDiceLoss(out5, src_label)
            loss4 = GeneralizedDiceLoss(out4, src_label)
            loss3 = GeneralizedDiceLoss(out3, src_label)
            loss2 = GeneralizedDiceLoss(out2, src_label)

            # loss = loss2 + loss3 + loss4 + loss5
            loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4
            
            loss.backward()
            self.optimizer_enc.step()
            self.optimizer_dec.step()

    def train(self):
        save_to_disk = self.local_rank == 0
        self.iteration = (self.start_epoch - 1) * len(self.train_loader)

        self.logger.info("#"*20 + " Start Attn Training " + "#"*20)

        self.encoder.train()
        self.decoder.train()

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_enc, 100, eta_min=0, last_epoch=-1)
        scheduler_enc = GradualWarmupScheduler(self.optimizer_enc, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_dec, 100, eta_min=0, last_epoch=-1)
        scheduler_dec = GradualWarmupScheduler(self.optimizer_dec, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
           
        for epoch in range(self.start_epoch, self.cfg.SOLVER.EPOCHS+1):
            self._train_epoch(epoch)
            scheduler_enc.step()
            scheduler_dec.step()