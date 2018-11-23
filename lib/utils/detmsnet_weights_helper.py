"""
Helper functions for converting resnet pretrained weights from other formats
"""
import os
import pickle

import torch

import nn as mynn
import utils.detectron_weight_helper as dwh
from core.config import cfg


def load_pretrained_imagenet_weights(model):
    """Load pretrained weights
    Args:
        num_layers: 50 for res50 and so on.
        model: the generalized rcnnn module
    """
    _, ext = os.path.splitext(cfg.MSNET.IMAGENET_PRETRAINED_WEIGHTS)
    model.init_weights(cfg.MSNET.IMAGENET_PRETRAINED_WEIGHTS)
