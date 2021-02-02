import argparse

import torch

from core.configs import cfg

from core.trainers.aspp_trainer import ASPPTrainer
from core.trainers.pranet_trainer import PraNetTrainer
from core.datasets.build import build_dataset

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def train(cfg, local_rank):
#     logger = logging.getLogger("FADA.trainer")
#     logger.info("Start training")

#     feature_extractor = build_feature_extractor(cfg)
#     device = torch.device(cfg.MODEL.DEVICE)
#     feature_extractor.to(device)
    
#     classifier = build_classifier(cfg)
#     classifier.to(device)

#     optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     optimizer_fea.zero_grad()
    
#     optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     optimizer_cls.zero_grad()

#     output_dir = cfg.OUTPUT_DIR
#     save_to_disk = local_rank == 0
#     iteration = 0

#     if cfg.resume:
#         logger.info("Loading checkpoint from {}".format(cfg.resume))
#         checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
#         model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
#         feature_extractor.load_state_dict(model_weights)
#         classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
#         classifier.load_state_dict(classifier_weights)
#         if "optimizer_fea" in checkpoint:
#             logger.info("Loading optimizer_fea from {}".format(cfg.resume))
#             optimizer.load(checkpoint['optimizer_fea'])
#         if "optimizer_cls" in checkpoint:
#             logger.info("Loading optimizer_cls from {}".format(cfg.resume))
#             optimizer.load(checkpoint['optimizer_cls'])
#         if "iteration" in checkpoint:
#             iteration = checkpoint['iteration']

#     src_train_data = build_dataset(cfg, mode='train', is_source=True)

#     train_loader = torch.utils.data.DataLoader(
#         src_train_data, 
#         batch_size=cfg.SOLVER.BATCH_SIZE, 
#         shuffle=True, 
#         num_workers=4, 
#         pin_memory=True, 
#         sampler=None, 
#         drop_last=True
#     )

#     criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

#     max_iters = cfg.SOLVER.MAX_ITER
#     logger.info("Start training")
#     meters = MetricLogger(delimiter="  ")
#     feature_extractor.train()
#     classifier.train()
#     start_training_time = time.time()
#     end = time.time()

#     for epoch in range(cfg.SOLVER.EPOCHS):
#         for i, (src_input, src_label, _) in enumerate(train_loader):
#             data_time = time.time() - end
#             current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
#             for index in range(len(optimizer_fea.param_groups)):
#                 optimizer_fea.param_groups[index]['lr'] = current_lr
#             for index in range(len(optimizer_cls.param_groups)):
#                 optimizer_cls.param_groups[index]['lr'] = current_lr*10

#             optimizer_fea.zero_grad()
#             optimizer_cls.zero_grad()
#             src_input = src_input.cuda(non_blocking=True)
#             src_label = src_label.cuda(non_blocking=True).long()
        
#             size = src_label.shape[-2:]
#             pred = classifier(feature_extractor(src_input), size)
        
#             loss = criterion(pred, src_label)
#             loss.backward()

#             optimizer_fea.step()
#             optimizer_cls.step()
#             meters.update(loss_seg=loss.item())
#             iteration+=1

#             batch_time = time.time() - end
#             end = time.time()
#             meters.update(time=batch_time, data=data_time)
#             eta_seconds = meters.time.global_avg * (max_iters - iteration)
#             eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#             if iteration % 20 == 0 or iteration == max_iters:
#                 logger.info(
#                     meters.delimiter.join(
#                         [
#                             "eta: {eta}",
#                             "iter: {iter}",
#                             "{meters}",
#                             "lr: {lr:.6f}",
#                             "max mem: {memory:.0f}",
#                         ]
#                     ).format(
#                         eta=eta_string,
#                         iter=iteration,
#                         meters=str(meters),
#                         lr=optimizer_fea.param_groups[0]["lr"],
#                         memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
#                     )
#                 )
    
#             if (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == cfg.SOLVER.STOP_ITER) and save_to_disk:
#                 filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
#                 torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(), 'classifier':classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_cls': optimizer_cls.state_dict()}, filename)
        
#             if iteration == max_iters:
#                 break
#             if iteration == cfg.SOLVER.STOP_ITER:
#                 break
    
#     total_training_time = time.time() - start_training_time
#     total_time_str = str(datetime.timedelta(seconds=total_training_time))
#     logger.info(
#         "Total training time: {} ({:.4f} s / it)".format(
#             total_time_str, total_training_time / (max_iters)
#         )
#     )

# def structure_loss(pred, mask):
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wbce + wiou).mean()

# def _train_epoch(train_loader, model, optimizer, epoch):
#     # ---- multi-scale training ----
#     size_rates = [0.75, 1, 1.25]
#     loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
#     for i, pack in enumerate(train_loader):
#         for rate in size_rates:
#             optimizer.zero_grad()
#             # ---- data prepare ----
#             images, gts, _ = pack
#             images = Variable(images).cuda()
#             gts = Variable(gts).cuda()
#             # ---- rescale ----
#             trainsize = int(round(opt.trainsize*rate/32)*32)
#             if rate != 1:
#                 images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
#                 gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
#             # ---- forward ----
#             lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
#             # ---- loss function ----
#             loss5 = structure_loss(lateral_map_5, gts)
#             loss4 = structure_loss(lateral_map_4, gts)
#             loss3 = structure_loss(lateral_map_3, gts)
#             loss2 = structure_loss(lateral_map_2, gts)
#             loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss
#             # ---- backward ----
#             loss.backward()
#             clip_gradient(optimizer, 0.5)
#             optimizer.step()
#             # ---- recording loss ----
#             if rate == 1:
#                 loss_record2.update(loss2.data, cfg.SOLVER.BATCH_SIZE)
#                 loss_record3.update(loss3.data, cfg.SOLVER.BATCH_SIZE)
#                 loss_record4.update(loss4.data, cfg.SOLVER.BATCH_SIZE)
#                 loss_record5.update(loss5.data, cfg.SOLVER.BATCH_SIZE)
#         # ---- train visualization ----
#         if i % 20 == 0 or i == len(train_loader):
#             print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
#                   '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
#                   format(datetime.now(), epoch, cfg.SOLVER.EPOCHS, i, len(train_loader),
#                          loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
#     save_path = cfg.OUTPUT_DIR
#     os.makedirs(save_path, exist_ok=True)
#     if (epoch+1) % 10 == 0:
#         torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
#         print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth'% epoch)

# def train(cfg):
#     model = PraNet().cuda()
#     model.train()

#     params = model.parameters()
#     optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR)

#     train_data = build_dataset(cfg, mode='train', is_source=True)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=None)

#     print("#"*20, "Start Training", "#"*20)
#     for epoch in range(cfg.SOLVER.EPOCHS):
#         _train_epoch(train_loader, model, optimizer, epoch)

def main(name, cfg, local_rank):
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    train_loader = torch.utils.data.DataLoader(
        src_train_data, 
        batch_size=cfg.SOLVER.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        sampler=None, 
        drop_last=True
    )

    if name == "aspp":
        trainer = ASPPTrainer(name, cfg, train_loader, local_rank)
    elif name == "pranet":
        trainer = PraNetTrainer(name, cfg, train_loader, local_rank)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    main("aspp", cfg, args.local_rank)