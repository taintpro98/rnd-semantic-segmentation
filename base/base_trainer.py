import torch
import os
import time
import sys 
from core.utils.utility import setup_logger

class BaseTrainer:
    def __init__(self, name, cfg, train_loader, local_rank):
        self.cfg = cfg
        self.logger = setup_logger(name, cfg.OUTPUT_DIR, local_rank)
        self.train_loader = train_loader
        self.local_rank = local_rank
        self.start_epoch = 1
        self.distributed = False
        # self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        if torch.cuda.is_available():
            self.with_cuda = True
            device = 'cuda'
            if torch.cuda.device_count() > 1:
                self.distributed = True
            torch.cuda.empty_cache()
        else:
            self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
            self.with_cuda = False
            device = 'cpu'

        self.device = torch.device(device)

        self.init_params()

        if cfg.resume:
            self.logger.info("Loading checkpoint from {}".format(self.cfg.resume))
            self._load_checkpoint()

    def init_params(self):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        raise NotImplementedError
        
    def _val_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_path):
        raise NotImplementedError
        
    def _load_checkpoint(self):
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