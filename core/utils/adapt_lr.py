import warnings
import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def adjust_learning_rate(method, base_lr, iters, max_iter, power):
    if method=='poly':
        lr = base_lr * ((1 - float(iters) / max_iter) ** (power))
    else:
        raise NotImplementedError
    return lr

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self,
        optimizer: Optimizer,
        T_max: int, eta_min: float = 0,
        last_epoch: int = -1,
        verbose=False,
        warmup_epochs=5
    ) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        offset_epoch = self.last_epoch - self.warmup_epochs
        if self.last_epoch < self.warmup_epochs:
            return [
                lr * (self.last_epoch + 1) / self.warmup_epochs
                for lr in self.base_lrs
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (offset_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * offset_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (offset_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]