import time
import datetime
import os 

import torch
import torch.nn.functional as F

from core.trainers.aspp_trainer import ASPPTrainer
from core.adapters.fada_adapter import FADAAdapter
from core.utils.utility import setup_logger, MetricLogger, soft_label_cross_entropy
from core.utils.adapt_lr import adjust_learning_rate

class AsppFada:
    def __init__(self, name, cfg, src_train_loader, tgt_train_loader, local_rank):
        self.cfg = cfg
        self.logger = setup_logger(name, cfg.OUTPUT_DIR, local_rank)
        self.aspp = ASPPTrainer(name, cfg, src_train_loader, local_rank, self.logger) #loaded checkpoints and src_train_loader
        self.fada = FADAAdapter(cfg, tgt_train_loader, self.aspp.device)
        if cfg.resume:
            self.fada._load_checkpoint(self.aspp.checkpoint, self.logger)

    def _save_checkpoint(self, adv_epoch, save_path):
        checkpoint = {
            'adv_epoch': adv_epoch, 
            'iteration': self.iteration, 
            'feature_extractor': self.aspp.feature_extractor.state_dict(), 
            'classifier': self.aspp.classifier.state_dict(), 
            'optimizer_fea': self.aspp.optimizer_fea.state_dict(), 
            'optimizer_cls': self.aspp.optimizer_cls.state_dict(), 
            'model_D': self.fada.model_D.state_dict(),
            'optimizer_D': self.fada.optimizer_D.state_dict()
        }
        torch.save(checkpoint, save_path)

    def train(self):
        save_to_disk = self.aspp.local_rank == 0
        self.iteration = (self.fada.start_adv_epoch - 1) * min(len(self.aspp.train_loader), len(self.fada.tgt_train_loader)) 

        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        bce_loss = torch.nn.BCELoss(reduction='none')

        max_iter = self.cfg.SOLVER.EPOCHS * min(len(self.aspp.train_loader), len(self.fada.tgt_train_loader))
        source_label = 0
        target_label = 1
        self.logger.info("#"*20 + " Start Adversarial Training " + "#"*20)
        meters = MetricLogger(delimiter="  ")

        self.aspp.feature_extractor.train()
        self.aspp.classifier.train()
        self.fada.model_D.train()

        start_training_time = time.time()
        end = time.time()

        for epoch in range(self.fada.start_adv_epoch, self.cfg.SOLVER.EPOCHS+1):
            for i, ((src_input, src_label, src_name), (tgt_input, _, _)) in enumerate(zip(self.aspp.train_loader, self.fada.tgt_train_loader)):
                data_time = time.time() - end
                
                self.iteration+=1
                current_lr = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR, self.iteration, max_iter, power=self.cfg.SOLVER.LR_POWER)
                current_lr_D = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR_D, self.iteration, max_iter, power=self.cfg.SOLVER.LR_POWER)
                for index in range(len(self.aspp.optimizer_fea.param_groups)):
                    self.aspp.optimizer_fea.param_groups[index]['lr'] = current_lr
                for index in range(len(self.aspp.optimizer_cls.param_groups)):
                    self.aspp.optimizer_cls.param_groups[index]['lr'] = current_lr*10
                for index in range(len(self.fada.optimizer_D.param_groups)):
                    self.fada.optimizer_D.param_groups[index]['lr'] = current_lr_D

                self.aspp.optimizer_fea.zero_grad()
                self.aspp.optimizer_cls.zero_grad()
                self.fada.optimizer_D.zero_grad()
                src_input = src_input.cuda(non_blocking=True)
                src_label = src_label.cuda(non_blocking=True).long()
                tgt_input = tgt_input.cuda(non_blocking=True)

                src_size = src_input.shape[-2:]
                tgt_size = tgt_input.shape[-2:]

                src_fea = self.aspp.feature_extractor(src_input)
                src_pred = self.aspp.classifier(src_fea, src_size) # float tensor B x C x H x W
                temperature = 1.8
                src_pred = src_pred.div(temperature)
                loss_seg = criterion(src_pred, src_label)
                loss_seg.backward()
                
                # generate soft labels
                src_soft_label = F.softmax(src_pred, dim=1).detach()
                src_soft_label[src_soft_label>0.9] = 0.9

                tgt_fea = self.aspp.feature_extractor(tgt_input)
                tgt_pred = self.aspp.classifier(tgt_fea, tgt_size)
                tgt_pred = tgt_pred.div(temperature)
                tgt_soft_label = F.softmax(tgt_pred, dim=1)

                tgt_soft_label = tgt_soft_label.detach()
                tgt_soft_label[tgt_soft_label>0.9] = 0.9

                tgt_D_pred = self.fada.model_D(tgt_fea, tgt_size)
                loss_adv_tgt = 0.001*soft_label_cross_entropy(tgt_D_pred, torch.cat((tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
                loss_adv_tgt.backward()

                self.aspp.optimizer_fea.step()
                self.aspp.optimizer_cls.step()

                self.fada.optimizer_D.zero_grad()

                src_D_pred = self.fada.model_D(src_fea.detach(), src_size)
                loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))
                loss_D_src.backward()

                tgt_D_pred = self.fada.model_D(tgt_fea.detach(), tgt_size)
                loss_D_tgt = 0.5*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label), dim=1))
                loss_D_tgt.backward()

                self.fada.optimizer_D.step()

                meters.update(loss_seg=loss_seg.item())
                meters.update(loss_adv_tgt=loss_adv_tgt.item())
                meters.update(loss_D=(loss_D_src.item()+loss_D_tgt.item()))
                meters.update(loss_D_src=loss_D_src.item())
                meters.update(loss_D_tgt=loss_D_tgt.item())

                n = src_input.size(0)

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)

                eta_seconds = meters.time.global_avg * (max_iter - self.iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if self.iteration % 20 == 0 or self.iteration == max_iter:
                    self.logger.info(
                        meters.delimiter.join(
                            [
                                "Epoch: {epoch}",
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                            ).format(
                                epoch=epoch,
                                eta=eta_string,
                                iter=self.iteration,
                                meters=str(meters),
                                lr=self.aspp.optimizer_fea.param_groups[0]["lr"],
                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            )
                        )
                    # self.logger.info('ETA: {}, Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}]', )
                
            if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 and save_to_disk:
                filename = os.path.join(self.cfg.OUTPUT_DIR, "AsppFada-{}.pth".format(epoch))
                self._save_checkpoint(epoch, filename)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f} s / epoch)".format(
                total_time_str, total_training_time / (self.cfg.SOLVER.EPOCHS)
            )
        )