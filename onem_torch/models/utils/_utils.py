from collections import OrderedDict
import warnings

import torch
from torch import nn
from torch.utils.model_zoo import load_url as _load_url


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a core

    It has a strong assumption that the modules have been registered
    into the core in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the core. So if `core` is passed, `core.feature1` can
    be returned, but not `core.feature1.layer2`.

    Arguments:
        model (nn.Module): core on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in core")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def load_state_dict_from_url(url, model_dir=None, progress=True, check_hash=False, file_name=None):
    """
    Loads the Torch serialized object at the given URL.
    
    If downloaded file is a zip file, it will be automatically
    decompressed.
    
    If the object is already present in `model_dir`, it's deserialized and
    returned.
    
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash (bool, optional): If True, the filename part of the URL should
            follow the naming convention ``filename-<sha256>.ext`` where ``<sha256>``
            is the first eight or more digits of the SHA256 hash of the contents of the file.
            The hash is used to ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): override the filename of the downloaded file.
            Default: None
    """
    # Issue warning to move data if old env is set
    if model_dir is None:
        torch.hub._validate_not_a_forked_repo()
        model_dir = torch.hub.get_dir()
    
    try:
        return _load_url(url, model_dir=model_dir, progress=progress, check_hash=check_hash, file_name=file_name)
    except Exception as e:
        warnings.warn(f"Failed to load from URL: {e}")
        raise
