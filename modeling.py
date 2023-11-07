from .utils import IntermediateLayerGetter
#from ._model import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception
)

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='ourmodelplus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='ourmodel':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = OurModel(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='ourmodelplus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='ouemodel':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = OurModel(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = xception.xception(pretrained= 'imagenet' if pretrained_backbone else False, replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 128
    
    if name=='ourmodelplus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='ourmodel':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = OurModel(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='ourmodelplus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='ourmodel':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = OurModel(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model



def ourmodel_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('ourmodel', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def ourmodel_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('ourmodel', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def ourmodel_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('ourmodel', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ourmodel_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('ourmodel', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ourmodel_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('ourmodel', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ourmodel_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    return _load_model('ourmodel', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def ourmodelplus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('ourmodelplus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def ourmodelplus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('ourmodelplus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def ourmodelplus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('ourmodelplus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def ourmodelplus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('ourmodelplus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def ourmodelplus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('ourmodelplus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def ourmodelplus_xception(num_classes=21, output_stride=8, pretrained_backbone=True):
    return _load_model('ourmodelplus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)