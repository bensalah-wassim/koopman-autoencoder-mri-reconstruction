'''
Koopman autoencoder architecture for k-space interpolation
author: Wassim Ben Salah
'''

# ==================================================
#           IMPORTS & PATH CONFIGURATION           #
# ==================================================
import os, sys, math
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



# ==================================================
#      COMPLEX ACTIVATIONS & INITIALISATIONS       #
# ==================================================

class ModReLU(nn.Module):
    """f(z) = ReLU(|z|+β)·z/|z|  — learnable threshold, can have dead neurons."""
    def __init__(self, n_channels, beta_init=0.0, device='cuda', **kw):
        super().__init__()
        self.beta = nn.Parameter(
            torch.full((1, n_channels), float(beta_init), device=device))
    def forward(self, z):
        mod = torch.abs(z)
        return torch.relu(mod + self.beta) / (mod + 1e-18) * z
    @staticmethod
    def init_gain(): return math.sqrt(2.0)


ACTIVATION_REGISTRY = {
    'modrelu':    ModReLU,
}

_INIT_SCHEME = {
    'modrelu':    'he',
}


def _complex_rayleigh_(weight, scheme, gain, seed=None):
    """
    Rayleigh modulus + uniform phase init (proper complex Gaussian).

    Glorot: s = gain·√(1/(fan_in+fan_out))  → E[|w|²] = 2s²
    He:     s = gain·√(1/fan_in)             → E[|w|²] = 2s²

    Rayleigh(s) + U(-π,π) = CN(0,2s²): maximum-entropy isotropic
    complex Gaussian — the correct prior for complex weights.
    """
    fan_out, fan_in = weight.shape
    s   = gain * math.sqrt(1.0 / (fan_in + fan_out if scheme == 'glorot' else fan_in))
    rng = np.random.RandomState(seed)
    mod = rng.rayleigh(scale=s, size=(fan_out, fan_in))
    phi = rng.uniform(-math.pi, math.pi, size=(fan_out, fan_in))
    w   = torch.complex(
        torch.tensor(mod * np.cos(phi), dtype=torch.float32),
        torch.tensor(mod * np.sin(phi), dtype=torch.float32),
    ).to(weight.device)
    with torch.no_grad():
        weight.copy_(w)


def init_complex_linear(module, activation_name='linear', is_latent=False, seed=None):
    """Activation-appropriate complex weight initialization."""
    name = activation_name.lower()
    
    if is_latent or name == 'linear':
        _complex_rayleigh_(module.weight, 'glorot', gain=1.0, seed=seed)
    else:
        scheme = _INIT_SCHEME.get(name, 'glorot')
        gain   = ACTIVATION_REGISTRY[name].init_gain()
        _complex_rayleigh_(module.weight, scheme, gain=gain, seed=seed)
        
    if module.bias is not None:
        with torch.no_grad():
            module.bias.data.zero_()


def build_activation(name, n_channels, device='cuda', **kwargs):
    name = name.lower()
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Available: {list(ACTIVATION_REGISTRY)}")
    try:
        return ACTIVATION_REGISTRY[name](n_channels=n_channels, device=device, **kwargs)
    except TypeError:
        return ACTIVATION_REGISTRY[name]()


# ==================================================
#                KOOPMAN AUTOENCODER               #
# ==================================================

class cKAE(nn.Module):
    """
    Complex Koopman AutoEncoder.

        Input (n1)
          ↓  Encoder: Linear(n1→n2) + activation
          ↓  Latent:  Linear(n2→n2) — Koopman operator approximation
          ↓  Decoder: Linear(n2→n1)
        Output (n1)

    activation_name options
    -----------------------
    'modrelu'     — learnable activation

    """
    def __init__(self, n1, n2=None, activation_name='modrelu',
                 beta_init=0.0, bias=False, init_seed=18,
                 device='cuda', dtype=torch.complex64):
        super().__init__()
        self.n1              = n1
        self.n2              = n2 if n2 is not None else n1
        self.activation_name = activation_name.lower()
        self.init_seed       = init_seed

        act_kw = {'beta_init': beta_init}
        dev    = str(device)


        self.Encoder = nn.Sequential(
            nn.Linear(self.n1, self.n2, bias=bias, device=device, dtype=dtype),
            build_activation(activation_name, self.n2, device=dev, **act_kw),
        )

        self.Latent = nn.Linear(self.n2, self.n2, bias=False, device=device, dtype=dtype)

        self.Decoder = nn.Sequential(
            nn.Linear(self.n2, self.n1, bias=bias, device=device, dtype=dtype),
        )

        self._init_weights()

    def _init_weights(self):
        init_complex_linear(self.Encoder[0], self.activation_name,
                            is_latent=False, seed=self.init_seed)
        
        init_complex_linear(self.Latent,     'linear',
                            is_latent=True,  seed=self.init_seed + 1)
        
        init_complex_linear(self.Decoder[0], 'linear',
                            is_latent=False, seed=self.init_seed + 2)

    def forward(self, x):
        return self.Decoder(self.Latent(self.Encoder(x)))
