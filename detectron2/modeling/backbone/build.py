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
    logger.debug(backbone.bottom_up.stem)
    return backbone
