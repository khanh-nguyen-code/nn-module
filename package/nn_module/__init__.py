from typing import Callable, Optional
import torch
from torch import nn


class Module(nn.Module):
    """
    Module: nn.Module with device and dtype properties
    """

    def __init__(self):
        super(Module, self).__init__()
        self.register_buffer("dummy_buffer", torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_buffer.device

    @property
    def dtype(self) -> torch.dtype:
        return self.dummy_buffer.dtype


class Functional(Module):
    """
    Functional: wrapper for function
    """

    def __init__(self, f: Callable, name: Optional[str] = None):
        super(Functional, self).__init__()
        self.f = f
        if name is None:
            self.name = f
        else:
            self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name}"

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)
