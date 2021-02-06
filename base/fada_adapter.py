import torch 

from core.models.build import build_adversarial_discriminator

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

class FADAAdapter:
    def __init__(self, cfg, tgt_train_loader, device):
        self.cfg = cfg
        self.device = device
        # self.src_train_loader = src_train_loader
        self.tgt_train_loader = tgt_train_loader

        # self.local_rank = local_rank
        self.start_adv_epoch = 1
        self.distributed = False
        # self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        self.init_params()

    def init_params(self):
        self.model_D = build_adversarial_discriminator(self.cfg)
        self.model_D.to(self.device)

        self.optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=self.cfg.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
        self.optimizer_D.zero_grad()

    def _load_checkpoint(self, checkpoint, logger):
        
        if "model_D" in checkpoint:
            logger.info("Loading model_D from {}".format(self.cfg.resume))
            model_D_weights = checkpoint['model_D'] if self.distributed else strip_prefix_if_present(checkpoint['model_D'], 'module.')
            self.model_D.load_state_dict(model_D_weights)
        if "adv_epoch" in checkpoint:
            self.start_adv_epoch = checkpoint['adv_epoch'] + 1