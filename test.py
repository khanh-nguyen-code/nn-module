import torch
from torch import nn

from package.nn_module import Module


class TestModule(Module):
    def __init__(self):
        super().__init__()
        # sub-module
        self.register_module("test_module", nn.Linear(
            in_features=1,
            out_features=1,
            dtype=torch.float32,
        ))
        # buffer: not in self.parameters()
        self.register_buffer("test_buffer", torch.rand(1, dtype=torch.float32))
        # parameter: in self.parameters()
        self.register_parameter("test_param", nn.Parameter(torch.empty(1, dtype=torch.float32)))

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)


if __name__ == "__main__":
    m = TestModule()
    print(m)
    print(list(m.buffers()))
    print(list(m.parameters()))
    print(m.test_module)
    print(m.test_buffer)
    print(m.test_param)
