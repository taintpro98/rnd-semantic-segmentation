
import os
import torch
import datetime
import time 
import logging

from core.utils.utility import MetricLogger
from core.models.build import build_model, build_feature_extractor, build_classifier, adjust_learning_rate

from base.base_trainer import BaseTrainer

class ASPPTrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, local_rank):
        super(ASPPTrainer, self).__init__(name, cfg, train_loader, local_rank)

    def train(self):
        self.logger.info("#"*20 + " Start Training " + "#"*20)

        feature_extractor = build_feature_extractor(self.cfg)
        # device = torch.device(self.cfg.MODEL.DEVICE)
        device = torch.device(self.device)
        feature_extractor.to(device)
    
        classifier = build_classifier(self.cfg)
        classifier.to(device)

        optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=self.cfg.SOLVER.BASE_LR, momentum=self.cfg.SOLVER.MOMENTUM, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        optimizer_fea.zero_grad()
    
        optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=self.cfg.SOLVER.BASE_LR*10, momentum=self.cfg.SOLVER.MOMENTUM, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        optimizer_cls.zero_grad()

        output_dir = self.cfg.OUTPUT_DIR
        save_to_disk = self.local_rank == 0
        iteration = 0

        if self.cfg.resume:
            self.logger.info("Loading checkpoint from {}".format(self.cfg.resume))
            checkpoint = torch.load(self.cfg.resume, map_location=self.device)
            model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
            feature_extractor.load_state_dict(model_weights)
            classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
            classifier.load_state_dict(classifier_weights)
            if "optimizer_fea" in checkpoint:
                self.logger.info("Loading optimizer_fea from {}".format(self.cfg.resume))
                optimizer.load(checkpoint['optimizer_fea'])
            if "optimizer_cls" in checkpoint:
                self.logger.info("Loading optimizer_cls from {}".format(self.cfg.resume))
                optimizer.load(checkpoint['optimizer_cls'])
            if "iteration" in checkpoint:
                iteration = checkpoint['iteration']

        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        max_iters = self.cfg.SOLVER.MAX_ITER
        self.logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        feature_extractor.train()
        classifier.train()
        start_training_time = time.time()
        end = time.time()

        for epoch in range(self.cfg.SOLVER.EPOCHS):
            for i, (src_input, src_label, _) in enumerate(self.train_loader):
                data_time = time.time() - end
                current_lr = adjust_learning_rate(self.cfg.SOLVER.LR_METHOD, self.cfg.SOLVER.BASE_LR, iteration, max_iters, power=self.cfg.SOLVER.LR_POWER)
                for index in range(len(optimizer_fea.param_groups)):
                    optimizer_fea.param_groups[index]['lr'] = current_lr
                for index in range(len(optimizer_cls.param_groups)):
                    optimizer_cls.param_groups[index]['lr'] = current_lr*10

                optimizer_fea.zero_grad()
                optimizer_cls.zero_grad()
                src_input = src_input.cuda(non_blocking=True)
                src_label = src_label.cuda(non_blocking=True).long()
        
                size = src_label.shape[-2:]
                pred = classifier(feature_extractor(src_input), size)
        
                loss = criterion(pred, src_label)
                loss.backward()

                optimizer_fea.step()
                optimizer_cls.step()
                meters.update(loss_seg=loss.item())
                iteration+=1

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)
                eta_seconds = meters.time.global_avg * (max_iters - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if iteration % 20 == 0 or iteration == max_iters:
                    self.logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer_fea.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
    
                if (iteration % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == self.cfg.SOLVER.STOP_ITER) and save_to_disk:
                    filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(), 'classifier':classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_cls': optimizer_cls.state_dict()}, filename)
        
                if iteration == max_iters:
                    break
                if iteration == self.cfg.SOLVER.STOP_ITER:
                    break
    
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iters)
            )
        )
