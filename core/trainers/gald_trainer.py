import os
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F

from base.base_trainer import BaseTrainer
from core.models.classifiers.gcpacc.gcpa_cc2 import GCPADecoder, GCPAEncoder
from core.utils.adapt_lr import CosineAnnealingWarmupLR, GradualWarmupScheduler

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, H, W) -> (C, N * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, H, W) -> (C, N, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, H, W) -> (C, N * H * W)
    return transposed.reshape(C, -1)

def GeneralizedDiceLoss(output, target, eps=1e-5, weight_type='square', ignore_label=255):
    """
        :param output: tensor (N, C, H, W) with not yet softmax score
        :param target: tensor (N, C, H, W) with one-hot column or (N, H, W) with labels
    """
    output = flatten(output) # [N, C, H, W] -> [C, N * H * W]
    output = F.softmax(output, dim=0)
    num_classes = output.size(0)

    if target.dim() == 3:
        # target shape [N, H, W]
        target = target.view(-1) # [N * H * W]
        tmp = torch.ones_like(torch.empty(target.size(0)))
        tmp[target == ignore_label] = 0
        tmp = torch.stack([tmp] * num_classes, dim=0)

        # tmp = np.ones(target.size(0)).astype(int) # numpy array [N*H*W] with all one
        # tmp[target.cpu() == ignore_label] = 0
        # tmp = np.concatenate([[tmp]] * num_classes, axis=0) # numpy array [C, N*H*W]
        # tmp = torch.from_numpy(tmp).cuda()
        output = output * tmp.cuda()

        target[target == ignore_label] = num_classes # convert ignore label 255 to max label
        target = F.one_hot(target, num_classes+1).permute(1, 0)[:-1, ...] # [C, N*H*W]
        
    else:
        target = flatten(target) # [N, C, H, W] -> [C, N * H * W] 

    target_sum = target.sum(-1) # [C, ]
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps) # [C, ]
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps) # [C, ]
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps) # [C, ]
    else:
        raise ValueError('Check out the weight_type: ', weight_type)

    intersect = (output * target).sum(-1) # [C, ]
    intersect_sum = (intersect * class_weights).sum() # scalar  
    denominator = (output * output + target * target).sum(-1) # [C, ]
    denominator_sum = (denominator * class_weights).sum() + eps #scalar

    # for i in range(output.size(0)):
    #     lossi = 2 * intersect[i] / (denominator[i] + eps)
    
    # logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum  

class GALDTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank, logger=None):
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
        for i, (src_input, src_label, _) in enumerate(self.train_loader):
            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()

            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            hardnetout = self.encoder(src_input)
            out5, out4, out3, out2 = self.decoder(src_input, hardnetout)

            # loss5 = GeneralizedDiceLoss(out5, src_label)
            # loss4 = GeneralizedDiceLoss(out4, src_label)
            # loss3 = GeneralizedDiceLoss(out3, src_label)
            # loss2 = GeneralizedDiceLoss(out2, src_label)

            loss5 = self.criterion(out5, src_label)
            loss4 = self.criterion(out4, src_label)
            loss3 = self.criterion(out3, src_label)
            loss2 = self.criterion(out2, src_label)

            # loss = loss2 + loss3 + loss4 + loss5
            loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4
            
            loss.backward()
            self.optimizer_enc.step()
            self.optimizer_dec.step()

            self.iteration+=1

            if i % 20 == 0 or i == len(self.train_loader):
                self.logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: [{:0.4f}], encode_learning_rate: [{:0.8f}], decode_learning_rate: [{:0.8f}]'.format(datetime.now(), epoch, self.cfg.SOLVER.EPOCHS, i, len(self.train_loader), loss.item(), self.optimizer_enc.param_groups[0]['lr'], self.optimizer_dec.param_groups[0]['lr']))

        save_path = self.cfg.OUTPUT_DIR
        os.makedirs(save_path, exist_ok=True)
        if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            self._save_checkpoint(epoch, save_path + 'Gald-%d.pth' % epoch)
            self.logger.info('[Saving Snapshot:] ' + save_path + 'Gald-{}.pth'.format(epoch))

    def train(self):
        save_to_disk = self.local_rank == 0
        self.iteration = (self.start_epoch - 1) * len(self.train_loader)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.logger.info("#"*20 + " Start Gald Training " + "#"*20)

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