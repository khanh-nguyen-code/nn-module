import math
import random

import torch
import torch_scatter
from torch import Tensor, nn

from package.nn_module import Module


def _sparse_multihead_softmax(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    :param values: (b, m, h)
    :param index: (m,)
    :return: (b, m, h)
    """
    assert len(values.shape) == 3
    assert len(index.shape) == 1
    assert index.dtype == torch.int64
    assert values.shape[1] == index.shape[0]

    values = values - values.max()  # shift does not change softmax, use this for less numerical errr
    exp_logits = torch.exp(values)
    exp_logits_sum = torch_scatter.scatter_sum(src=exp_logits, index=index, dim=1)[:, index, :]
    return exp_logits / (exp_logits_sum + 1e-16)


def _sparse_multihead_attention(
        q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor, head: torch.Tensor,
):
    """
    :param q: (b, n, d)
    :param k: (b, n, d)
    :param mask: (2, m)
    :param head: (d,)
    :return: (b, m)
    """
    assert len(q.shape) == 3
    assert len(k.shape) == 3
    assert q.shape == k.shape

    assert len(mask.shape) == 2
    assert mask.shape[0] == 2
    assert mask.dtype == torch.int64

    assert len(head.shape) == 1
    assert head.shape[0] == q.shape[2]

    src_q = q[:, mask[0, :], :]  # (b, m, d)
    dst_k = k[:, mask[1, :], :]  # (b, m, d)
    a = torch_scatter.scatter_sum(src=src_q * dst_k, index=head, dim=2)
    return a


class SparseMultiheadAttention(Module):
    def __init__(self,
                 hidden_dim: int,
                 query_dim: int,
                 key_dim: int,
                 mask: torch.Tensor,
                 num_heads: int = 1,
                 ):
        """
        :param mask: (2, m)
        """
        assert hidden_dim % num_heads == 0
        assert len(mask.shape) == 2
        assert mask.shape[0] == 2
        assert mask.dtype == torch.int64

        super().__init__()
        self.name = f"{self.__class__.__name__}(hidden_dim={hidden_dim}, query_dim={query_dim}, key_dim={key_dim}, num_heads={num_heads})"

        self.register_parameter("w_q", nn.Parameter(torch.empty(query_dim, hidden_dim, dtype=torch.float32)))
        self.register_parameter("w_k", nn.Parameter(torch.empty(key_dim, hidden_dim, dtype=torch.float32)))

        self.register_buffer("mask", mask)
        self.register_buffer("head", torch.tensor(
            [i // (hidden_dim // num_heads) for i in range(hidden_dim)]
            , dtype=torch.int64,
        ))
        self._reset_parameters()

    def __repr__(self) -> str:
        return self.name

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)

    def forward(self, x_q: Tensor, x_k: Tensor, reduce_dim: int = 1) -> torch.Tensor:
        """
        :param x_q: (b, n, query_dim)
        :param x_k: (b, n, key_dim)
        :param reduce_dim:
        :return: (b, m, num_heads) - reduce_dim=1, sum of each row is 1
        """
        assert len(x_q.shape) == 3
        assert len(x_k.shape) == 3
        assert x_q.shape[:2] == x_k.shape[:2]
        assert x_q.shape[2] == self.w_q.shape[0]
        assert x_k.shape[2] == self.w_k.shape[0]
        q = x_q @ self.w_q  # (b, n, hidden_dim)
        k = x_k @ self.w_k  # (b, n, hidden_dim)
        a_values = _sparse_multihead_attention(
            q=q,
            k=k,
            mask=self.mask,
            head=self.head,
        ) / math.sqrt(self.w_k.shape[0])  # (b, m, num_heads)
        a_values = _sparse_multihead_softmax(values=a_values, index=self.mask[1 - reduce_dim, :])
        return a_values


if __name__ == "__main__":
    def _get_values_from_matrix(matrix: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """
        :param matrix: (b, n1, n2)
        :param edge: (2, m)
        :return: (b, m)
        """
        assert len(matrix.shape) == 3
        assert len(edge.shape) == 2
        assert edge.dtype == torch.int64
        assert edge.shape[0] == 2

        out = []
        for m_i in range(edge.shape[1]):
            i, j = edge[:, m_i]
            out.append(matrix[:, i, j])
        return torch.stack(out).T


    def _attention(q: torch.Tensor, k: torch.Tensor):
        """
        :param q: (b, n, d)
        :param k: (b, n, d)
        :return: (b, n, n)
        """
        assert len(q.shape) == 3
        assert len(k.shape) == 3
        assert q.shape == k.shape
        a = torch.bmm(q, k.transpose(1, 2))
        return a


    # test sparse_multi_headed_attention
    b, n, d = 3, 5, 4
    q = torch.rand(b, n, d)
    k = torch.rand(b, n, d)
    mask = []
    for i in range(n):
        for j in range(n):
            if random.random() < 0.5:
                mask.append((i, j))
    mask = torch.tensor(mask).T
    head = torch.tensor([0, 0, 1, 1])
    expected = torch.stack([
        _get_values_from_matrix(_attention(q[:, :, 0:2], k[:, :, 0:2]), mask),
        _get_values_from_matrix(_attention(q[:, :, 2:4], k[:, :, 2:4]), mask),
    ], dim=-1)
    actual = _sparse_multihead_attention(q, k, mask, head)
    assert torch.isclose(expected, actual).all()
