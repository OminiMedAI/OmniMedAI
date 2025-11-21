# -*- coding: UTF-8 -*-

import os
from typing import Optional

import numpy as np

try:
    import nibabel
except ImportError:
    nibabel = None

try:
    import nrrd
except ImportError:
    nrrd = None


def image_loader_3d(impath: str, root='', index_order='F') -> Optional[np.ndarray]:
    """
    Args:
        impath: image path
        root: Where impath is relative path, use root to concat impath
        index_order: {'C', 'F'}, optional
        Specifies the index order of the resulting data array. Either 'C' (C-order) where the dimensions are ordered from
        slowest-varying to fastest-varying (e.g. (z, y, x)), or 'F' (Fortran-order) where the dimensions are ordered
        from fastest-varying to slowest-varying (e.g. (x, y, z)).

    Returns:

    """
    assert index_order in ['F', 'C']
    impath = os.path.join(root, impath)
    if impath and os.path.exists(impath):
        if impath.endswith('.nrrd'):
            if nrrd is None:
                raise ImportError("nrrd package is required to read .nrrd files. Install with: pip install pynrrd")
            nrrd_data, _ = nrrd.read(impath, index_order=index_order)
            return nrrd_data
        elif impath.endswith('.nii.gz') or impath.endswith('.nii'):
            if nibabel is None:
                raise ImportError("nibabel package is required to read .nii files. Install with: pip install nibabel")
            image = nibabel.load(impath).get_data()
            if index_order == 'C':
                image = np.transpose(image, [2, 1, 0])
            return image
    else:
        return None
