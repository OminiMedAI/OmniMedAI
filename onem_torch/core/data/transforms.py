# -*- coding: UTF-8 -*-

from torchvision.transforms import transforms

try:
    from monai.transforms import (
        AddChannel,
        Compose,
        RandRotate90,
        Resize,
        ScaleIntensity,
        EnsureType,
    )
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

__all__ = ['create_standard_image_transformer']


def create_standard_image_transformer(input_size, phase='train', normalize_method='imagenet', is_nii: bool = False,
                                      **kwargs):
    """Standard image transformer.

    :param input_size: The core's input image size.
    :param phase: phase of transformer, train or valid or test supported.
    :param normalize_method: Normalize method, imagenet or -1+1 supported.
    :param is_nii: 是不是多通过nii，当成2d来训练
    :return:
    """
    assert phase in ['train', 'valid', 'test'], "`phase` not found, only 'train', 'valid', 'test' supported!"
    normalize = {'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                 '-1+1': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]}
    assert normalize_method in normalize, "`normalize_method` not found, only 'imagenet', '-1+1' supported!"
    if not is_nii:
        if phase == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                # transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize[normalize_method])])
        else:
            return transforms.Compose([
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize[normalize_method])])
    else:
        if not HAS_MONAI:
            raise ImportError("MONAI package is required for medical imaging transforms. Install with: pip install monai")
        roi_size = kwargs.get('roi_size', [3, 96, 96])
        if phase == 'train':
            return Compose([ScaleIntensity(), AddChannel(), Resize(roi_size), EnsureType()])
        else:
            return Compose([ScaleIntensity(), AddChannel(), Resize(roi_size), EnsureType()])
