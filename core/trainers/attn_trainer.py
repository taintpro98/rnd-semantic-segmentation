import os
from datetime import datetime

import torch
import torch.nn.functional as F

from base.base_trainer import BaseTrainer
from core.models.classifiers.attn.eff import Encoder, Decoder, AttnEfficientNetUnet
from core.models.classifiers.attn.loss import TverskyLoss, CompoundLoss, BinaryCrossEntropyLoss, MultiscaleLoss
from core.utils.adapt_lr import GradualWarmupScheduler
from core.utils.utility import generate_scales, probs_to_onehot

class AttnTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank, logger=None):
        super(AttnTrainer, self).__init__(name, cfg, train_loader, local_rank, logger)

    def init_params(self):
        self.encoder = Encoder(backbone_name="efficientnet-b2")
        self.decoder = Decoder(backbone_name="efficientnet-b2")
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.optimizer_enc = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.SOLVER.BASE_LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=self.cfg.SOLVER.BASE_LR*10, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

    def _train_epoch(self, epoch):
        for i, (src_input, src_label, _) in enumerate(self.train_loader):

            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()

            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            endpoints = self.encoder(src_input)
            outputs = self.decoder(endpoints)

            if self.cfg.MODEL.NUM_CLASSES > 1:
                label = F.one_hot(src_label.squeeze(dim=1), self.cfg.MODEL.NUM_CLASSES).permute(0, 3, 1, 2).float()
            scaled_labels = generate_scales(label, self.decoder.output_scales)

            output = outputs[0]
            pred_probs = torch.sigmoid(output)  # B x C x H x W
            pred_probs = probs_to_onehot(pred_probs)

            loss = self.criterion(outputs, scaled_labels)
            loss.backward()

            self.optimizer_enc.step()
            self.optimizer_dec.step()

            self.iteration+=1

            if i % 20 == 0 or i == len(self.train_loader):
                self.logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: [{:0.4f}], encode_learning_rate: [{:0.8f}], decode_learning_rate: [{:0.8f}]'.format(datetime.now(), epoch, self.cfg.SOLVER.EPOCHS, i, len(self.train_loader), loss.item(), self.optimizer_enc.param_groups[0]['lr'], self.optimizer_dec.param_groups[0]['lr']))

        save_path = self.cfg.OUTPUT_DIR
        os.makedirs(save_path, exist_ok=True)
        if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            self._save_checkpoint(epoch, save_path + 'Attn-%d.pth' % epoch)
            self.logger.info('[Saving Snapshot:] ' + save_path + 'Attn-{}.pth'.format(epoch))

    def train(self):
        save_to_disk = self.local_rank == 0
        self.iteration = (self.start_epoch - 1) * len(self.train_loader)

        self.criterion = MultiscaleLoss(
            CompoundLoss([
                TverskyLoss(),
                BinaryCrossEntropyLoss()
            ])
        )

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

    def _save_checkpoint(self, epoch, save_path):
        checkpoint = {
            'epoch': epoch,
            'iteration': self.iteration,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer_enc': self.optimizer_enc.state_dict(),
            'optimizer_dec': self.optimizer_dec.state_dict()
        }
        torch.save(checkpoint, save_path)

    def _load_checkpoint(self):
        self.checkpoint = torch.load(self.cfg.resume, map_location=self.device)
        self.encoder.load_state_dict(self.checkpoint['encoder'])
        self.decoder.load_state_dict(self.checkpoint['decoder'])
        if "optimizer_enc" in self.checkpoint:
            self.logger.info("Loading encoder optimizer from {}".format(self.cfg.resume))
            self.optimizer_enc.load_state_dict(self.checkpoint['optimizer_enc'])
        if "optimizer_dec" in self.checkpoint:
            self.logger.info("Loading decoder optimizer from {}".format(self.cfg.resume))
            self.optimizer_dec.load_state_dict(self.checkpoint['optimizer_dec'])
        if "iteration" in self.checkpoint:
            self.iteration = self.checkpoint['iteration']
        if "epoch" in self.checkpoint:
            self.start_epoch = self.checkpoint['epoch'] + 1


class AttnWrapTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank, logger=None):
        super(AttnWrapTrainer, self).__init__(name, cfg, train_loader, local_rank, logger)

    def init_params(self):
        self.model = AttnEfficientNetUnet(backbone_name="efficientnet-b2")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.SOLVER.BASE_LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

    def _train_epoch(self, epoch):
        for i, (src_input, src_label, _) in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True).long()

            outputs = self.model(src_input)

            if self.cfg.MODEL.NUM_CLASSES > 1:
                label = F.one_hot(src_label.squeeze(dim=1), self.cfg.MODEL.NUM_CLASSES).permute(0, 3, 1, 2).float()
            scaled_labels = generate_scales(label, self.model.decoder.output_scales)

            output = outputs[0]
            pred_probs = torch.sigmoid(output)  # B x C x H x W
            pred_probs = probs_to_onehot(pred_probs)

            loss = self.criterion(outputs, scaled_labels)
            loss.backward()

            self.optimizer.step()

            self.iteration+=1

            if i % 20 == 0 or i == len(self.train_loader):
                self.logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: [{:0.4f}], learning_rate: [{:0.8f}]'.format(datetime.now(), epoch, self.cfg.SOLVER.EPOCHS, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))

        save_path = self.cfg.OUTPUT_DIR
        os.makedirs(save_path, exist_ok=True)
        if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            self._save_checkpoint(epoch, save_path + 'AttnWrap-%d.pth' % epoch)
            self.logger.info('[Saving Snapshot:] ' + save_path + 'AttnWrap-{}.pth'.format(epoch))

    def train(self):
        save_to_disk = self.local_rank == 0
        self.iteration = (self.start_epoch - 1) * len(self.train_loader)

        self.criterion = MultiscaleLoss(
            CompoundLoss([
                TverskyLoss(),
                BinaryCrossEntropyLoss()
            ])
        )

        self.logger.info("#"*20 + " Start Attn Training " + "#"*20)

        self.model.train()

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100, eta_min=0, last_epoch=-1)
        scheduler = GradualWarmupScheduler(self.optimizer, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)

        for epoch in range(self.start_epoch, self.cfg.SOLVER.EPOCHS+1):
            self._train_epoch(epoch)
            scheduler.step()

    def _save_checkpoint(self, epoch, save_path):
        checkpoint = {
            'epoch': epoch,
            'iteration': self.iteration,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)

    def _load_checkpoint(self):
        self.checkpoint = torch.load(self.cfg.resume, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model'])
        if "optimizer" in self.checkpoint:
            self.logger.info("Loading optimizer from {}".format(self.cfg.resume))
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        if "iteration" in self.checkpoint:
            self.iteration = self.checkpoint['iteration']
        if "epoch" in self.checkpoint:
            self.start_epoch = self.checkpoint['epoch'] + 1