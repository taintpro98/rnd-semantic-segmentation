#!/usr/bin/env python
# coding: utf-8

import torch
import logging
import os
from model import FOTSLoss
import time

class BaseTrainer:
    def __init__(self, config, model, resume_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.config = config
        self.optimizer = self.model.optimize(config['optimizer_type'], config['optimizer'])
        self.epochs = config['trainer']['epochs']
        self.val_interval = self.config["validation"]["val_interval"]
        self.start_epoch = 1
        self.checkpoint_dir = config["trainer"]["save_dir"]

        # if torch.cuda.is_available():
        #     if config['cuda']:
        #         self.with_cuda = True
        #         self.gpus = {i: item for i, item in enumerate(self.config['gpus'])}
        #         device = 'cuda'
        #         if torch.cuda.device_count() > 1 and len(self.gpus) > 1:
        #             self.model.parallelize()
        #         torch.cuda.empty_cache()
        #     else:
        #         self.with_cuda = False
        #         device = 'cpu'
        # else:
        #     self.logger.warning('Warning: There\'s no CUDA support on this machine, '
        #                         'training is performed on CPU.')
        #     self.with_cuda = False
        #     device = 'cpu'

        if torch.cuda.is_available():
            self.with_cuda = True
            device = 'cuda'
        else:
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
            self.with_cuda = False
            device = 'cpu'

        self.device = torch.device(device)
        self.loss = FOTSLoss(config, device)
        self.model.to(self.device)
        if resume_path:
            print('resume-------------')
            self._load_checkpoint(resume_path)
        
    def _train_epoch(self, epoch):
        raise NotImplementedError
        
    def _val_epoch(self, epoch):
        raise NotImplementedError
        
    # def _log_memory_useage(self):
    #     if not self.with_cuda: return

    #     template = """Memory Usage: \n{}"""
    #     usage = []
    #     for deviceID, device in self.gpus.items():
    #         deviceID = int(deviceID)
    #         allocated = torch.cuda.memory_allocated(deviceID) / (1024 * 1024)
    #         cached = torch.cuda.memory_cached(deviceID) / (1024 * 1024)

    #         usage.append('    CUDA: {}  Allocated: {} MB Cached: {} MB \n'.format(device, allocated, cached))

    #     content = ''.join(usage)
    #     content = template.format(content)
        
    def train(self):
        best_score = None
        for epoch in range(self.start_epoch, self.epochs + 1):

            tic = time.time()
            print("Training on epoch: {}/{}".format(epoch, self.epochs))
            try:
                train_log = self._train_epoch(epoch)
                print('Epoch {}, Loss: {:.6f}, There are F1 Score: {:.6f}, Precision: {:.6f}, Recall: {:.6f}, Accuracy by char: {:.6f}, Accuracy by field: {:.6f}'.format(
                    epoch,
                    train_log['loss'],
                    train_log['f1'],
                    train_log['precision'],
                    train_log['recall'],
                    train_log['epoch_accuracy_char'],
                    train_log['epoch_accuracy_field']
                ))
            except torch.cuda.CudaError:
                print("error cuda")
                # self._log_memory_useage()
            if epoch % self.val_interval == 0 or epoch == self.epochs:
                val_log = self._val_epoch(epoch)
                print('Validation Epoch: {}, F1 Score: {:.6f}, Precision: {:.6f}, Recall: {:.6f}'.format(epoch, val_log['val_f1'], val_log['val_precision'], val_log['val_recall']))
                if best_score is None or val_log['val_f1'] > best_score:
                    self.log = {**train_log, **val_log}
                    self._save_checkpoint(epoch)
                    best_score = val_log['val_f1']

                
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.log,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'.format(epoch, self.log['loss']))
        torch.save(state, filename)
        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        
        
    def _load_checkpoint(self, resume_path):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(torch.device('cuda'))
        self.log = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))



