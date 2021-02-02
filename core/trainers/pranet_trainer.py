import os 
from datetime import datetime

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from base.base_trainer import BaseTrainer
from core.models.classifiers.pranet.PraNet_Res2Net import PraNet
from core.utils.utils import AvgMeter, clip_gradient
from core.utils.adapt_lr import GradualWarmupScheduler

class PraNetTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank):
        super(PraNetTrainer, self).__init__(name, cfg, train_loader, local_rank)
        
    def init_params(self):
        self.trainsize = self.cfg.INPUT.TRAINSIZE
        self.model = PraNet().cuda()
        params = self.model.parameters()
        self.optimizer = torch.optim.Adam(params, self.cfg.SOLVER.BASE_LR/8)

    def structure_loss(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def _train_epoch(self, epoch):
        # ---- multi-scale training ----
        size_rates = [0.75, 1, 1.25]
        loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, pack in enumerate(self.train_loader):
            for rate in size_rates:
                self.optimizer.zero_grad()
                # ---- data prepare ----
                images, gts, _ = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(self.trainsize*rate/32)*32)
                if rate != 1:
                    images = F.upsample(images, size=(self.trainsize, self.trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(self.trainsize, self.trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = self.model(images)
                # ---- loss function ----
                loss5 = self.structure_loss(lateral_map_5, gts)
                loss4 = self.structure_loss(lateral_map_4, gts)
                loss3 = self.structure_loss(lateral_map_3, gts)
                loss2 = self.structure_loss(lateral_map_2, gts)
                loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
                # ---- backward ----
                loss.backward()
                clip_gradient(self.optimizer, 0.5)
                self.optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record2.update(loss2.data, self.cfg.SOLVER.BATCH_SIZE)
                    loss_record3.update(loss3.data, self.cfg.SOLVER.BATCH_SIZE)
                    loss_record4.update(loss4.data, self.cfg.SOLVER.BATCH_SIZE)
                    loss_record5.update(loss5.data, self.cfg.SOLVER.BATCH_SIZE)
            # ---- train visualization ----
            if i % 20 == 0 or i == len(self.train_loader):
                self.logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}, learning_rate: {:0.8f}]'.
                    format(datetime.now(), epoch, self.cfg.SOLVER.EPOCHS, i, len(self.train_loader),
                             loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show(), self.optimizer.param_groups[0]['lr'] ))
        save_path = self.cfg.OUTPUT_DIR
        os.makedirs(save_path, exist_ok=True)
        if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            self._save_checkpoint(epoch, save_path + 'PraNet-%d.pth' % epoch)
            self.logger.info('[Saving Snapshot:] ' + save_path + 'PraNet-{}.pth'.format(epoch))

    def _save_checkpoint(self, epoch, save_path):
        checkpoint = {
            'epoch': epoch, 
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)

    def _load_checkpoint(self):
        self.logger.info("Loading checkpoint from {}".format(self.cfg.resume))
        self.checkpoint = torch.load(self.cfg.resume, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model'])
        if "optimizer" in self.checkpoint:
            self.logger.info("Loading optimizer from {}".format(self.cfg.resume))
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        if "epoch" in self.checkpoint:
            self.start_epoch = self.checkpoint['epoch'] + 1

    def train(self):
        self.model.train()
        self.logger.info("#"*20 + " Start Training " + "#"*20)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100, eta_min=0, last_epoch=-1)
        scheduler = GradualWarmupScheduler(self.optimizer, multiplier=8, total_epoch=5, after_scheduler=cosine_scheduler)
        for epoch in range(self.start_epoch, self.cfg.SOLVER.EPOCHS+1):
            # adjust_lr(self.optimizer, self.cfg.SOLVER.BASE_LR, epoch, self.cfg.SOLVER.DECAY_RATE, self.cfg.SOLVER.DECAY_EPOCH)
            self._train_epoch(epoch)
            scheduler.step()