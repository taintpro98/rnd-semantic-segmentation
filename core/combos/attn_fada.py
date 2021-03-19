import time
import datetime
import os 

import torch
import torch.nn.functional as F

from core.utils.utility import setup_logger, soft_label_cross_entropy, MetricLogger, generate_scales
from core.trainers.attn_trainer import AttnTrainer, AttnWrapTrainer
from core.adapters.fada_adapter import FADAAdapter
from core.utils.adapt_lr import adjust_learning_rate, GradualWarmupScheduler, CosineAnnealingWarmupLR
from core.models.classifiers.attn.loss import TverskyLoss, CompoundLoss, BinaryCrossEntropyLoss, MultiscaleLoss

class AttnFada:
    def __init__(self, name, cfg, src_train_loader, tgt_train_loader, local_rank):
        self.cfg = cfg
        self.logger = setup_logger(name, cfg.OUTPUT_DIR, local_rank)
        self.attn = AttnTrainer(name, cfg, src_train_loader, local_rank, self.logger)
        self.fada = FADAAdapter(cfg, tgt_train_loader, self.attn.device)
        if cfg.resume:
            self.fada._load_checkpoint(self.attn.checkpoint, self.logger)

    def _save_checkpoint(self, adv_epoch, save_path):
        checkpoint = {
            'adv_epoch': adv_epoch, 
            'iteration': self.iteration, 
            'encoder': self.attn.encoder.state_dict(), 
            'decoder': self.attn.decoder.state_dict(), 
            'optimizer_enc': self.attn.optimizer_enc.state_dict(), 
            'optimizer_dec': self.attn.optimizer_dec.state_dict(), 
            'model_D': self.fada.model_D.state_dict(),
            'optimizer_D': self.fada.optimizer_D.state_dict()
        }
        torch.save(checkpoint, save_path)

    def train(self):
        save_to_disk = self.attn.local_rank == 0
        self.iteration = (self.fada.start_adv_epoch - 1) * min(len(self.attn.train_loader), len(self.fada.tgt_train_loader)) 

        # criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        # bce_loss = torch.nn.BCELoss(reduction='none')

        criterion = MultiscaleLoss(
            CompoundLoss([
                TverskyLoss(),
                BinaryCrossEntropyLoss()
            ])
        )

        max_iter = self.cfg.SOLVER.EPOCHS * min(len(self.attn.train_loader), len(self.fada.tgt_train_loader))
        source_label = 0
        target_label = 1
        self.logger.info("#"*20 + " Start Adversarial Training " + "#"*20)
        meters = MetricLogger(delimiter="  ")

        self.attn.encoder.train()
        self.attn.decoder.train()
        self.fada.model_D.train()

        start_training_time = time.time()
        end = time.time()

        scheduler_enc = CosineAnnealingWarmupLR(self.attn.optimizer_enc, T_max=50, warmup_epochs=5)
        scheduler_dec = CosineAnnealingWarmupLR(self.attn.optimizer_dec, T_max=50, warmup_epochs=5)
        scheduler_D = CosineAnnealingWarmupLR(self.fada.optimizer_D, T_max=50, warmup_epochs=5)

        for epoch in range(self.fada.start_adv_epoch, self.cfg.SOLVER.EPOCHS+1):
            for i, ((src_input, src_label, src_name), (tgt_input, _, _)) in enumerate(zip(self.attn.train_loader, self.fada.tgt_train_loader)):
                data_time = time.time() - end
                
                self.iteration+=1
                # current_lr = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR, self.iteration, max_iter, power=self.cfg.SOLVER.LR_POWER)
                # current_lr_D = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR_D, self.iteration, max_iter, power=self.cfg.SOLVER.LR_POWER)
                # for index in range(len(self.attn.optimizer_enc.param_groups)):
                #     self.attn.optimizer_enc.param_groups[index]['lr'] = current_lr
                # for index in range(len(self.attn.optimizer_dec.param_groups)):
                #     self.attn.optimizer_dec.param_groups[index]['lr'] = current_lr*10
                for index in range(len(self.fada.optimizer_D.param_groups)):
                    self.fada.optimizer_D.param_groups[index]['lr'] = current_lr_D

                self.attn.optimizer_enc.zero_grad()
                self.attn.optimizer_dec.zero_grad()
                self.fada.optimizer_D.zero_grad()

                src_input = src_input.cuda(non_blocking=True)
                src_label = src_label.cuda(non_blocking=True).long() # tensor B x 1 x H x W
                src_label = src_label.squeeze(1) # tensor B x H x W
                tgt_input = tgt_input.cuda(non_blocking=True)

                src_size = src_input.shape[-2:]
                tgt_size = tgt_input.shape[-2:]

                if self.cfg.MODEL.NUM_CLASSES > 1:
                    label = F.one_hot(src_label.squeeze(dim=1), self.cfg.MODEL.NUM_CLASSES).permute(0, 3, 1, 2).float()
                scaled_labels = generate_scales(label, self.attn.decoder.output_scales)

                src_endpoints = self.attn.encoder(src_input)
                src_outputs = self.attn.decoder(src_endpoints)
                src_output = src_outputs[0]
                src_output = torch.sigmoid(src_output) # B x C x H x W

                temperature = 1.8
                src_output = src_output.div(temperature)
                loss_seg = criterion(src_outputs, scaled_labels)
                loss_seg.backward()

                # generate soft labels
                src_soft_label = F.softmax(src_output, dim=1).detach()
                src_soft_label[src_soft_label>0.9] = 0.9

                tgt_endpoints = self.attn.encoder(tgt_input)
                tgt_outputs = self.attn.decoder(tgt_endpoints)
                tgt_output = tgt_outputs[0]
                tgt_output = torch.sigmoid(tgt_output) # B x C x H x W
                tgt_output = tgt_output.div(temperature)

                tgt_soft_label = F.softmax(tgt_output, dim=1)
                tgt_soft_label = tgt_soft_label.detach()
                tgt_soft_label[tgt_soft_label>0.9] = 0.9

                tgt_D_output = self.fada.model_D(tgt_endpoints["reduction_5"] , tgt_size)
                loss_adv_tgt = 0.001*soft_label_cross_entropy(tgt_D_output, torch.cat((tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
                loss_adv_tgt.backward()

                self.attn.optimizer_enc.step()
                self.attn.optimizer_dec.step()

                self.fada.optimizer_D.zero_grad()

                src_D_pred = self.fada.model_D(src_endpoints["reduction_5"].detach(), src_size)
                loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))
                loss_D_src.backward()

                tgt_D_output = self.fada.model_D(tgt_endpoints["reduction_5"].detach(), tgt_size)
                loss_D_tgt = 0.5*soft_label_cross_entropy(tgt_D_output, torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label), dim=1))
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
                                lr=self.attn.optimizer_enc.param_groups[0]["lr"],
                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            )
                        )

            if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 and save_to_disk:
                filename = os.path.join(self.cfg.OUTPUT_DIR, "AttnFada-{}.pth".format(epoch))
                self._save_checkpoint(epoch, filename)
            scheduler_enc.step()
            scheduler_dec.step()
            scheduler_D.step()

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f} s / epoch)".format(
                total_time_str, total_training_time / (self.cfg.SOLVER.EPOCHS)
            )
        )