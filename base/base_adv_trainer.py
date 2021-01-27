from base.base_trainer import BaseTrainer

class FADATrainer(BaseTrainer):
    def __init__(self, name, cfg, train_loader, tgt_loader, local_rank):
        super(BaseTrainer, self).__init__(name, cfg, train_loader, local_rank)
        self.tgt_loader = tgt_loader
        