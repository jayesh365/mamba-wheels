# s4d parts from this repo: https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py

"""Utility nn components, in particular handling activations, initializations, and normalization layers."""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X



"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # C = torch.randn(H, N // 2, dtype=torch.cfloat)
        C = torch.randn(H, N // 2, dtype=torch.float)
        # self.C = nn.Parameter(torch.view_as_real(C))
        self.C = nn.Parameter(C)
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        # A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)

        # changed to test Behnoush hypothesis
        # self.register_buffer("A_imag", A_imag)
        # self.register("A_imag", A_imag, lr)



    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        # C = self.C # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)
        # A = -torch.exp(self.log_A_real) # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        # K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K))

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified




class S4DTokenClassifier(nn.Module):
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
    def transposed(self) -> bool:  # noqa: D102
        return self._transposed

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
        lr: float = None,
        transposed: bool = True,
        prenorm: bool = False,
        bias: bool = True,
        embed: bool = False
    ):
        super().__init__()
        self._d_model = d_model
        self._d_state = d_state
        self._dropout = dropout
        self._n_vocab = n_vocab
        self._n_layers = n_layers
        self._prenorm = prenorm
        self._transposed = transposed
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
            print("real embedding is being applied")
        else:
            if pad_token_index != n_vocab:
                raise ValueError(
                    "When using one-hot encoding, the padding token must always be a value equivalent to the vocabulary size."
                )

            self.embedding = partial(F.one_hot, num_classes=(n_vocab + 1))
            print("one-hot is being applied")
            self._d_model = pad_token_index

        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.s4d_layers.append(
                S4D(d_model=d_model, dropout=dropout, transposed=transposed, lr=lr)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.cl_head = nn.Linear(self.d_model, self.n_vocab, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        # if not self.embed:
        x = x.long().squeeze()
        x = self.embedding(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(
            self.s4d_layers, self.norms, self.dropout_layers
        ):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = dropout(z)
            x = x + z

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        x = self.cl_head(x)

        return x