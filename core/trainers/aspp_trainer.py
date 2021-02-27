import os
import torch
import datetime
import time 
import logging

from core.utils.utility import MetricLogger, strip_prefix_if_present
from core.models.build import build_model, build_feature_extractor, build_classifier
from core.utils.adapt_lr import adjust_learning_rate
from base.base_trainer import BaseTrainer

class ASPPTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank, logger=None):
        super(ASPPTrainer, self).__init__(name, cfg, train_loader, local_rank, logger)
        
    def init_params(self):
        self.feature_extractor = build_feature_extractor(self.cfg)
        self.feature_extractor.to(self.device)
    
        self.classifier = build_classifier(self.cfg)
        self.classifier.to(self.device)

        self.optimizer_fea = torch.optim.SGD(self.feature_extractor.parameters(), lr=self.cfg.SOLVER.BASE_LR, momentum=self.cfg.SOLVER.MOMENTUM, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        self.optimizer_fea.zero_grad()
    
        self.optimizer_cls = torch.optim.SGD(self.classifier.parameters(), lr=self.cfg.SOLVER.BASE_LR*10, momentum=self.cfg.SOLVER.MOMENTUM, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        self.optimizer_cls.zero_grad()

    def _load_checkpoint(self):
        self.checkpoint = torch.load(self.cfg.resume, map_location=self.device)
        
        model_weights = self.checkpoint['feature_extractor'] if self.distributed else strip_prefix_if_present(self.checkpoint['feature_extractor'], 'module.')
        self.feature_extractor.load_state_dict(model_weights)
        classifier_weights = self.checkpoint['classifier'] if self.distributed else strip_prefix_if_present(self.checkpoint['classifier'], 'module.')
        self.classifier.load_state_dict(classifier_weights)
        if "optimizer_fea" in self.checkpoint:
            self.logger.info("Loading optimizer_fea from {}".format(self.cfg.resume))
            self.optimizer_fea.load_state_dict(self.checkpoint['optimizer_fea'])
        if "optimizer_cls" in self.checkpoint:
            self.logger.info("Loading optimizer_cls from {}".format(self.cfg.resume))
            self.optimizer_cls.load_state_dict(self.checkpoint['optimizer_cls'])
        if "iteration" in self.checkpoint:
            self.iteration = self.checkpoint['iteration']
        if "epoch" in self.checkpoint:
            self.start_epoch = self.checkpoint['epoch'] + 1

    def _save_checkpoint(self, epoch, save_path):
        checkpoint = {
            'epoch': epoch, 
            'iteration': self.iteration, 
            'feature_extractor': self.feature_extractor.state_dict(), 
            'classifier': self.classifier.state_dict(), 
            'optimizer_fea': self.optimizer_fea.state_dict(), 
            'optimizer_cls': self.optimizer_cls.state_dict()
        }
        torch.save(checkpoint, save_path)

    def train(self):
        output_dir = self.cfg.OUTPUT_DIR
        save_to_disk = self.local_rank == 0
        self.iteration = (self.start_epoch - 1) * len(self.train_loader)     
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        max_iters = self.cfg.SOLVER.EPOCHS * len(self.train_loader)
        self.logger.info("#"*20 + " Start Training " + "#"*20)
        meters = MetricLogger(delimiter="  ")
        self.feature_extractor.train()
        self.classifier.train()
        start_training_time = time.time()
        end = time.time()

        for epoch in range(self.start_epoch, self.cfg.SOLVER.EPOCHS+1):
            for i, (src_input, src_label, _) in enumerate(self.train_loader):
                data_time = time.time() - end
                current_lr = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR, self.iteration, max_iters, power=self.cfg.SOLVER.LR_POWER)
                for index in range(len(self.optimizer_fea.param_groups)):
                    self.optimizer_fea.param_groups[index]['lr'] = current_lr
                for index in range(len(self.optimizer_cls.param_groups)):
                    self.optimizer_cls.param_groups[index]['lr'] = current_lr*10

                self.optimizer_fea.zero_grad()
                self.optimizer_cls.zero_grad()
                src_input = src_input.cuda(non_blocking=True)
                src_label = src_label.cuda(non_blocking=True).long()
        
                size = src_label.shape[-2:]
                pred = self.classifier(self.feature_extractor(src_input), size)
        
                loss = criterion(pred, src_label)
                loss.backward()

                self.optimizer_fea.step()
                self.optimizer_cls.step()
                meters.update(loss_seg=loss.item())
                self.iteration+=1

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)
                eta_seconds = meters.time.global_avg * (max_iters - self.iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.iteration % 20 == 0 or self.iteration == max_iters:
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
                            lr=self.optimizer_fea.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
    
            if epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 and save_to_disk:
                filename = os.path.join(output_dir, "Aspp-{}.pth".format(epoch))
                self._save_checkpoint(epoch, filename)
        
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f} s / epoch)".format(
                total_time_str, total_training_time / (self.cfg.SOLVER.EPOCHS)
            )
        )