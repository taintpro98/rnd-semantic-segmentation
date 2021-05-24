import time
import datetime
import os 

import torch
import torch.nn.functional as F

from core.utils.utility import setup_logger, soft_label_cross_entropy, MetricLogger, generate_scales, GeneralizedDiceLoss, dump_json
from core.trainers.gald_trainer import GALDTrainer
from core.adapters.fada_adapter import FADAAdapter
from core.utils.adapt_lr import adjust_learning_rate, GradualWarmupScheduler, CosineAnnealingWarmupLR

class GaldFada:
    def __init__(self, name, cfg, src_train_loader, tgt_train_loader, local_rank):
        self.cfg = cfg
        self.logger = setup_logger(name + "_train", cfg.OUTPUT_DIR, local_rank)
        self.gald = GALDTrainer(name, cfg, src_train_loader, local_rank, self.logger)
        self.fada = FADAAdapter(cfg, tgt_train_loader, self.gald.device)
        if cfg.resume:
            self.fada._load_checkpoint(self.gald.checkpoint, self.logger)

        self.lr_data = list()
        self.D_lr_data = list()
        self.loss_seg_data = list()
        self.loss_adv_tgt_data = list()
        self.loss_D_src_data = list()
        self.loss_D_tgt_data = list()

    def _save_checkpoint(self, adv_epoch, save_path):
        checkpoint = {
            'adv_epoch': adv_epoch, 
            'iteration': self.iteration, 
            'encoder': self.gald.encoder.state_dict(), 
            'decoder': self.gald.decoder.state_dict(), 
            'optimizer_enc': self.gald.optimizer_enc.state_dict(), 
            'optimizer_dec': self.gald.optimizer_dec.state_dict(), 
            'model_D': self.fada.model_D.state_dict(),
            'optimizer_D': self.fada.optimizer_D.state_dict()
        }
        torch.save(checkpoint, save_path)

    def train(self):
        save_to_disk = self.gald.local_rank == 0
        self.iteration = (self.fada.start_adv_epoch - 1) * min(len(self.gald.train_loader), len(self.fada.tgt_train_loader)) 

        # criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        # bce_loss = torch.nn.BCELoss(reduction='none')

        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        max_iter = self.cfg.SOLVER.EPOCHS * min(len(self.gald.train_loader), len(self.fada.tgt_train_loader))
        source_label = 0
        target_label = 1
        self.logger.info("#"*20 + " Start Adversarial Training " + "#"*20)
        meters = MetricLogger(delimiter="  ")

        self.gald.encoder.train()
        self.gald.decoder.train()
        self.fada.model_D.train()

        start_training_time = time.time()
        end = time.time()

        # scheduler_enc = CosineAnnealingWarmupLR(self.attn.optimizer_enc, T_max=50, warmup_epochs=5)
        # scheduler_dec = CosineAnnealingWarmupLR(self.attn.optimizer_dec, T_max=50, warmup_epochs=5)
        # scheduler_D = CosineAnnealingWarmupLR(self.fada.optimizer_D, T_max=50, warmup_epochs=5)

        for epoch in range(self.fada.start_adv_epoch, self.cfg.SOLVER.EPOCHS+1):
            for i, ((src_input, src_label, src_name), (tgt_input, _, _)) in enumerate(zip(self.gald.train_loader, self.fada.tgt_train_loader)):
                data_time = time.time() - end
                
                self.iteration+=1
                current_lr = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR, self.iteration, max_iter, power=self.cfg.SOLVER.LR_POWER)
                current_lr_D = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR_D, self.iteration, max_iter, power=self.cfg.SOLVER.LR_POWER)
                for index in range(len(self.gald.optimizer_enc.param_groups)):
                    self.gald.optimizer_enc.param_groups[index]['lr'] = current_lr
                for index in range(len(self.gald.optimizer_dec.param_groups)):
                    self.gald.optimizer_dec.param_groups[index]['lr'] = current_lr*10
                for index in range(len(self.fada.optimizer_D.param_groups)):
                    self.fada.optimizer_D.param_groups[index]['lr'] = current_lr_D

                self.gald.optimizer_enc.zero_grad()
                self.gald.optimizer_dec.zero_grad()
                self.fada.optimizer_D.zero_grad()

                src_input = src_input.cuda(non_blocking=True)
                src_label = src_label.cuda(non_blocking=True).long()
                tgt_input = tgt_input.cuda(non_blocking=True)

                src_size = src_input.shape[-2:]
                tgt_size = tgt_input.shape[-2:]


                src_hardnetout = self.gald.encoder(src_input)
                src_outputs = self.gald.decoder(src_input, src_hardnetout)

                src_output = src_outputs[-1] # float tensor B x C x H x W
                # src_output = F.softmax(src_output) 
                temperature = 1.8
                src_output = src_output.div(temperature)
                # loss_seg = criterion(src_output, src_label)
                loss_seg = GeneralizedDiceLoss(src_output, src_label)
                loss_seg.backward()

                # generate soft labels
                src_soft_label = F.softmax(src_output, dim=1).detach()
                src_soft_label[src_soft_label>0.9] = 0.9

                tgt_hardnetout = self.gald.encoder(tgt_input)
                tgt_outputs = self.gald.decoder(tgt_input, tgt_hardnetout)
                tgt_output = tgt_outputs[-1]
                # tgt_output = torch.sigmoid(tgt_output) # B x C x H x W
                tgt_output = tgt_output.div(temperature)

                tgt_soft_label = F.softmax(tgt_output, dim=1)
                tgt_soft_label = tgt_soft_label.detach()
                tgt_soft_label[tgt_soft_label>0.9] = 0.9

                tgt_D_output = self.fada.model_D(tgt_hardnetout[3] , tgt_size)
                loss_adv_tgt = 0.001*soft_label_cross_entropy(tgt_D_output, torch.cat((tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
                loss_adv_tgt.backward()

                self.gald.optimizer_enc.step()
                self.gald.optimizer_dec.step()

                self.fada.optimizer_D.zero_grad()

                src_D_pred = self.fada.model_D(src_hardnetout[3].detach(), src_size)
                loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))
                loss_D_src.backward()

                tgt_D_output = self.fada.model_D(tgt_hardnetout[3].detach(), tgt_size)
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

                self.lr_data.append(self.gald.optimizer_enc.param_groups[0]["lr"])
                self.D_lr_data.append(self.fada.optimizer_D.param_groups[0]['lr'])
                self.loss_seg_data.append(loss_seg.item())
                self.loss_adv_tgt_data.append(loss_adv_tgt.item())
                self.loss_D_src_data.append(loss_D_src.item())
                self.loss_D_tgt_data.append(loss_D_tgt.item())

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
                                lr=self.gald.optimizer_enc.param_groups[0]["lr"],
                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            )
                        )

            if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 and save_to_disk:
                filename = os.path.join(self.cfg.OUTPUT_DIR, "GaldFada-{}.pth".format(epoch))
                self._save_checkpoint(epoch, filename)
            # scheduler_enc.step()
            # scheduler_dec.step()
            # scheduler_D.step()

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f} s / epoch)".format(
                total_time_str, total_training_time / (self.cfg.SOLVER.EPOCHS)
            )
        )

        mydata = {
            "learning rate": self.lr_data,
            "discriminator learning rate": self.D_lr_data,
            "segmentation loss": self.loss_seg_data,
            "target adversarial loss": self.loss_adv_tgt_data,
            "source discriminator loss": self.loss_D_src_data,
            "target discriminator loss": self.loss_D_tgt_data
        }
        json_path = os.path.join(self.cfg.OUTPUT_DIR, "gald_fada_chart_params.json")
        dump_json(json_path, mydata)