import torch
from mamba_ssm import Mamba
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial


class Mamba_model(nn.Module):
    """S4D Token Classifier."""

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def d_state(self) -> int:  # noqa: D102
        return self._d_state

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def prenorm(self) -> bool:  # noqa: D102
        return self._prenorm

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(  # noqa: D107
        self,
        d_model: int | None,
        d_state: int,
        dropout: float,
        n_layers: int,
        n_vocab: int,
        pad_token_index: int | None = None,
        prenorm: bool = False,
        bias: bool = True,
        embed: bool = False,
    ):
        super().__init__()
        self._d_model = d_model
        self._d_state = d_state
        self._dropout = dropout
        self._n_vocab = n_vocab
        self._n_layers = n_layers
        self._prenorm = prenorm
        self._bias = bias
        self.embed = embed

        if pad_token_index == None:
            pad_token_index = n_vocab

        self._pad_token_index = pad_token_index
        if embed:
            if pad_token_index is not None:
                assert pad_token_index == n_vocab, "We want pad token to be equal to the vocab size"
                n_vocab += 1
            self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_model, padding_idx=pad_token_index)
        else:
            if pad_token_index != n_vocab:
                raise ValueError(
                    "When using one-hot encoding, the padding token must always be a value equivalent to the vocabulary size."
                )

            self.embedding = partial(F.one_hot, num_classes=(n_vocab+1))
            self._d_model = pad_token_index

        self.mamba_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.mamba_layers.append(
                Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2).to("cuda")
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.cl_head = nn.Linear(self.d_model, self.n_vocab, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = x.long().squeeze()
        x = self.embedding(x)

        for layer, norm, dropout in zip(
            self.mamba_layers, self.norms, self.dropout_layers
        ):
            z = x
            if self.prenorm:
                z = norm(z)

            z = layer(z)
            z = dropout(z)
            x = x + z

            if not self.prenorm:
                x = norm(x)#.transpose(-1, -2)).transpose(-1, -2)

        # x = x.transpose(-1, -2)
        x = self.cl_head(x)

        return x