from core.models.feature_extractor import resnet_feature_extractor
from core.models.classifiers.aspp.classifier import ASPP_Classifier_V2
from .discriminator import * 

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
        raise NotImplementedError
    return backbone

def build_classifier(cfg):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('vgg'):
        classifier = ASPP_Classifier_V2(1024, [6, 12, 18, 24], [6, 12, 18, 24], cfg.MODEL.NUM_CLASSES)
    elif backbone_name.startswith('resnet'):
        classifier = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], cfg.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError
    return classifier

def build_adversarial_discriminator(cfg, num_features=None, mid_nc=256):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('vgg'):
        if num_features is None:
            num_features = 1024
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    elif backbone_name.startswith('resnet'):
        if num_features is None:
            num_features = 2048
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    elif backbone_name.startswith('efficientnet'):
        if num_features is None:
            num_features = 1408
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    elif backbone_name.startswith('hardnet'):
        if num_features is None:
            num_features = 1024
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError
    return model_D