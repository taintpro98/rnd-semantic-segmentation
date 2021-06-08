import os
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F

from base.base_trainer import BaseTrainer
from core.models.classifiers.gcpacc.gcpa_cc2 import GCPADecoder, GCPAEncoder
from core.utils.adapt_lr import CosineAnnealingWarmupLR, GradualWarmupScheduler, adjust_learning_rate
from core.utils.utility import dump_json

class GALDTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank, logger=None):
        super(GALDTrainer, self).__init__(name, cfg, train_loader, local_rank, logger)

    def init_params(self):
        self.encoder = GCPAEncoder()
        self.decoder = GCPADecoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.optimizer_enc = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.SOLVER.BASE_LR)
        self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=self.cfg.SOLVER.BASE_LR*10)

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

    def _train_epoch(self, epoch):
        max_iter = self.cfg.SOLVER.EPOCHS * len(self.train_loader)
        for i, (src_input, src_label, _) in enumerate(self.train_loader):
            current_lr = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR, self.iteration, max_iter, power=self.cfg.SOLVER.LR_POWER)
            for index in range(len(self.optimizer_enc.param_groups)):
                self.optimizer_enc.param_groups[index]['lr'] = current_lr
            for index in range(len(self.optimizer_dec.param_groups)):
                self.optimizer_dec.param_groups[index]['lr'] = current_lr*10
            
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

            self.lr_data.append(self.optimizer_enc.param_groups[0]["lr"])
            self.loss_data.append(loss.item())

            if i % 20 == 0 or i == len(self.train_loader):
                self.logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: [{:0.4f}], encode_learning_rate: [{:0.8f}], decode_learning_rate: [{:0.8f}]'.format(datetime.now(), epoch, self.cfg.SOLVER.EPOCHS, i, len(self.train_loader), loss.item(), self.optimizer_enc.param_groups[0]['lr'], self.optimizer_dec.param_groups[0]['lr']))

        save_path = self.cfg.OUTPUT_DIR
        os.makedirs(save_path, exist_ok=True)
        if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            self._save_checkpoint(epoch, os.path.join(save_path, 'Gald-%d.pth' % epoch))
            self.logger.info('[Saving Snapshot:] ' + save_path + 'Gald-{}.pth'.format(epoch))

    def train(self):
        save_to_disk = self.local_rank == 0
        self.iteration = (self.start_epoch - 1) * len(self.train_loader)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.logger.info("#"*20 + " Start Gald Training " + "#"*20)

        self.encoder.train()
        self.decoder.train()

        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_enc, 100, eta_min=0, last_epoch=-1)
        # scheduler_enc = GradualWarmupScheduler(self.optimizer_enc, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_dec, 100, eta_min=0, last_epoch=-1)
        # scheduler_dec = GradualWarmupScheduler(self.optimizer_dec, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
           
        for epoch in range(self.start_epoch, self.cfg.SOLVER.EPOCHS+1):
            self._train_epoch(epoch)
            # scheduler_enc.step()
            # scheduler_dec.step()
        mydata = {
            "learning rate": self.lr_data,
            "loss": self.loss_data
        }
        json_path = os.path.join(self.cfg.OUTPUT_DIR, "gald_chart_params.json")
        dump_json(json_path, mydata)