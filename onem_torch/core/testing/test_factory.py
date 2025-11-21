# -*- coding: UTF-8 -*-

import torch.nn as nn

from ..factory.losses_factory import create_losses
from ..factory.model_factory import create_model
from ..factory.optimizer_factory import create_optimizer, create_lr_scheduler


def test_create_model():
    create_model('inception_v3', num_classes=100, pretrained=False)
    create_model('Inception3', num_classes=100)
    create_model('densenet161', pretrained=False)
    create_model('detection.keypointrcnn_resnet50_fpn', pretrained_backbone=False)
    try:
        create_model('mobilenet', pretrained=False)
    except ValueError:
        pass


def test_create_optimizer():
    model = create_model('alexnet')
    create_optimizer('RMSprop', model.parameters())
    create_optimizer('RMSprop', [{'params': model.features.parameters(), 'lr': 0.01},
                                 {'params': model.classifier.parameters(), 'lr': 0.001}],
                     alpha=0.99)


def test_create_lr_scheduler():
    model = create_model('alexnet')
    rms = create_optimizer('RMSprop', [{'params': model.features.parameters(), 'lr': 0.01},
                                       {'params': model.classifier.parameters(), 'lr': 0.001}],
                           alpha=0.99)
    create_lr_scheduler('cosine', rms, T_max=10)


def test_create_losses():
    assert isinstance(create_losses('l1', reduction='mean'), nn.L1Loss)
    losses = create_losses([{'loss': 'softmax_ce'}, {'loss': 'l1'}])
    assert isinstance(losses, list) and isinstance(losses[0], nn.CrossEntropyLoss) and isinstance(losses[1], nn.L1Loss)
    try:
        create_losses('l1', Error_parm='mean')
    except TypeError:
        pass

    try:
        create_losses('lxxx')
    except ValueError:
        pass

    try:
        create_losses([{'loss': 'softmax_ce', 'reduction': 'mean'}, {'loss': 'l1'}])
    except AssertionError:
        pass
