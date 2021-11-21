# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from .backbone import Backbone

import logging
logger = logging.getLogger(__name__)

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

It must returns an instance of :class:`Backbone`.
"""


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape)
    assert isinstance(backbone, Backbone)

    # Debug loggine
    breakpoint()
    import inspect

    logger.debug(input_shape)
    logger.debug(inspect.getsourcelines(BACKBONE_REGISTRY._obj_map.get(backbone_name)))
    logger.debug(backbone._out_feature_channels)
    logger.debug(backbone._out_feature_strides)
    logger.debug(backbone.bottom_up._out_feature_channels)
    logger.debug(backbone.bottom_up._out_feature_strides)

    # ResNet architecture
    # ResNet stem with convolutions
    logger.debug(backbone.bottom_up.stem)
    logger.debug('The out-channels from the stem are input to the next stage')
    # Batch normalization function for the stem convolutions
    logger.debug(backbone.bottom_up.stem.conv1_1.norm)
    # ResNet stage 2
    for bb in range(len(backbone.bottom_up.res2)):
        logger.debug(backbone.bottom_up.res2[bb])
    logger.debug('The first convolution layer and the shortcut layer both take the inputs from the stem out-channels')
    logger.debug('The output of the shortcut layer and the output of the last convolution layer in each stage (res2, res3, res4, etc.) have the same shape')
    for bb in range(len(backbone.bottom_up.res3)):
        logger.debug(backbone.bottom_up.res3[bb])
    for bb in range(len(backbone.bottom_up.res5)):
        logger.debug(backbone.bottom_up.res5[bb])

    return backbone
