from typing import Tuple, Optional, Dict

import torch
import torchdiffeq

from package.nn_module import Module


class ODEFunc(Module):
    """
    ODEFunc: dy/dt = f(t, y)
    """

    def on_before_integration(self, x0: torch.Tensor, t_list: torch.Tensor):
        pass

    def forward(self, t: float, y: torch.Tensor) -> torch.Tensor:
        raise Exception("not implemented")


class NeuralODE(Module):
    def __init__(
            self,
            ode_func: ODEFunc,
            t_list: torch.Tensor = torch.tensor([0.0, 1.0], dtype=torch.float32),
            adjoint: bool = True,
            ode_int_options: Optional[Dict] = None,
    ):
        """
        :param ode_func: f(t, y)
        """
        super().__init__()
        self.register_module("ode_func", ode_func)
        self.register_buffer("t_list", t_list)
        self.adjoint = adjoint
        self.ode_int_options = {}
        if ode_int_options is not None:
            self.ode_int_options.update(ode_int_options)

    def forward(self, y0: torch.Tensor) -> torch.Tensor:
        return self.integrate(y0)[-1]

    def integrate(self, y0: torch.Tensor) -> torch.Tensor:
        self.ode_func.on_before_integration(y0, self.t_list)
        if self.adjoint:
            ode_int = torchdiffeq.odeint_adjoint
        else:
            ode_int = torchdiffeq.odeint

        y_list = ode_int(self.ode_func, y0, self.t_list, **self.ode_int_options)
        return y_list
