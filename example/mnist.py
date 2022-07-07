import os.path
from typing import Callable, Iterable
import multiprocessing as mp
import torch
import torchvision
from torch import nn, optim

from package.nn_module.mlp import MLP
from package.nn_module.neuralode import ODEFunc, NeuralODE
from sklearn.model_selection import train_test_split

from package.nn_module.resnet import ResNet


def module_size(module: nn.Module) -> int:
    return sum([param.numel() for param in module.parameters()])


class TimeIndependentODEFunc(ODEFunc):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.register_module("module", module)

    def forward(self, t: float, y: torch.Tensor) -> torch.Tensor:
        return self.module.forward(y)


def train(
        module: nn.Module,
        optimizer_func: Callable[[Iterable[nn.Parameter]], optim.Optimizer],
        device: torch.device = torch.device("cpu"),
):
    dataset = torchvision.datasets.MNIST("./data/", download=True)
    X, y = dataset.data.flatten(start_dim=1), dataset.targets
    X, y = X.to(device).to(torch.float32), y.to(device)
    module.to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.01)

    def decode(X: torch.Tensor, y_actual: torch.Tensor) -> float:
        module.eval()
        with torch.no_grad():
            logits = module.forward(X)
            y_pred = torch.argmax(logits, dim=1)
        count = float((y_pred == y_actual).sum().detach().cpu().numpy())
        batch = y_actual.shape[0]
        return count / batch

    print(f"training {module}")
    print(f"model size {module_size(module)}")

    loss_func = nn.CrossEntropyLoss()
    optimizer = optimizer_func(module.parameters())
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        module.train()
        module.zero_grad()

        y_pred = module.forward(X_train)
        loss = loss_func.forward(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch == num_epochs:
            loss_value = float(loss.detach().cpu().numpy())
            print(f"epoch {epoch}, loss {loss_value}, accuracy {decode(X_test, y_test)}")


if __name__ == "__main__":
    torch.set_num_threads(mp.cpu_count())
    torch.set_num_interop_threads(mp.cpu_count())

    if not os.path.exists("./data"):
        os.mkdir("./data")

    train(
        module=MLP(
            dim_list=[28 * 28, 512, 512, 512, 10],
            activation=nn.ReLU,
        ),
        optimizer_func=lambda params: optim.Adam(params, lr=1e-3),
    )

    train(
        module=nn.Sequential(
            nn.Linear(
                in_features=28 * 28,
                out_features=256,
            ),
            ResNet(
                MLP(
                    dim_list=[256, 256, 256],
                    activation=nn.Tanh,
                ),
                MLP(
                    dim_list=[256, 256, 256],
                    activation=nn.Tanh,
                ),
                MLP(
                    dim_list=[256, 256, 256],
                    activation=nn.Tanh,
                ),
                MLP(
                    dim_list=[256, 256, 256],
                    activation=nn.Tanh,
                ),
                MLP(
                    dim_list=[256, 256, 256],
                    activation=nn.Tanh,
                ),
                MLP(
                    dim_list=[256, 256, 256],
                    activation=nn.Tanh,
                )
            ),
            nn.Linear(
                in_features=256,
                out_features=10,
            )
        ),
        optimizer_func=lambda params: optim.Adam(params, lr=1e-3),
    )

    train(
        module=nn.Sequential(
            nn.Linear(
                in_features=28 * 28,
                out_features=512,
            ),
            NeuralODE(
                ode_func=TimeIndependentODEFunc(
                    module=MLP(
                        dim_list=[512, 512, 512],
                        activation=nn.Tanh,
                    )
                ),
                t_list=torch.tensor([0.0, 0.1]),
                ode_int_options={
                    "rtol": 1e-7,
                    "atol": 1e-9,
                    "method": "implicit_adams",
                }
            ),
            nn.Linear(
                in_features=512,
                out_features=10,
            )
        ),
        optimizer_func=lambda params: optim.Adam(params, lr=1e-3),
    )
