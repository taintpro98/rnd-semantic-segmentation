def build_model(cfg):
    model_name, backbone_name = cfg.MODEL.NAME.split('_')
    if model_name=='deeplab':
        model = deeplab(backbone_name, cfg.MODEL.NUM_CLASSES, pretrained_weights=cfg.MODEL.WEIGHTS, freeze_bn=cfg.MODEL.FREEZE_BN)
    else:
        raise NotImplementedError
    return model

def build_feature_extractor(cfg):
    model_name, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('resnet'):
        backbone = resnet_feature_extractor(backbone_name, pretrained_weights=cfg.MODEL.WEIGHTS, aux=False, pretrained_backbone=True, freeze_bn=cfg.MODEL.FREEZE_BN)
    elif backbone_name.startswith('vgg'):
        backbone = vgg_feature_extractor(backbone_name, pretrained_weights=cfg.MODEL.WEIGHTS, aux=False, pretrained_backbone=True, freeze_bn=cfg.MODEL.FREEZE_BN)
    else:
        raise NotImplementError
    return backbone