import os

import torch

from base.base_trainer import BaseTrainer
from core.models.classifiers.attn.eff import Encoder, Decoder
from core.models.classifiers.attn.loss import TverskyLoss

class AttnTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank, logger=None):
        super(AttnTrainer, self).__init__(name, cfg, train_loader, local_rank, logger)

    def init_params(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.optimizer_enc = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.SOLVER.BASE_LR, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        self.optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=self.cfg.SOLVER.BASE_LR*10, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

    def _train_epoch(self, epoch):
        for i, (src_input, src_label, _) in enumerate(self.train_loader):

            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()

            endpoints = self.encoder(src_input)
            outputs = self.decoder(endpoints)

            if self.cfg.MODEL.NUM_CLASSES > 1:
                label = F.one_hot(src_label.squeeze(dim=1), self.cfg.MODEL.NUM_CLASSES).permute(0, 3, 1, 2).float()
            scaled_labels = generate_scales(label, self.decoder.output_scales)

            output = outputs[0]
            pred_probs = torch.sigmoid(output)  # B x C x H x W
            pred_probs = probs_to_onehot(pred_probs)

            loss = criterion(outputs, scaled_labels)
            loss.backward()

            self.optimizer_enc.step()
            self.optimizer_dec.step()

            self.iteration+=1

            if i % 20 = 0 or i == len(self.train_loader):
                self.logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], learning_rate: {:0.8f}]'.format())

        save_path = self.cfg.OUTPUT_DIR
        os.makedirs(save_path, exist_ok=True)
        if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            self._save_checkpoint(epoch, save_path + 'Attn-%d.pth' % epoch)
            self.logger.info('[Saving Snapshot:] ' + save_path + 'Attn-{}.pth'.format(epoch))

    def train(self):
        output_dir = self.cfg.OUTPUT_DIR
        save_to_disk = self.local_rank == 0
        self.iteration = (self.start_epoch - 1) * len(self.train_loader)

        criterion = TverskyLoss()

        self.logger.info("#"*20 + " Start Attn Training " + "#"*20)

        self.encoder.train()
        self.decoder.train()

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_enc, 100, eta_min=0, last_epoch=-1)
        scheduler_enc = GradualWarmupScheduler(self.optimizer_enc, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_dec, 100, eta_min=0, last_epoch=-1)
        scheduler_dec = GradualWarmupScheduler(self.optimizer_dec, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)

        for epoch in range(self.start_epoch, self.cfg.SOLVER.EPOCH+1):
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