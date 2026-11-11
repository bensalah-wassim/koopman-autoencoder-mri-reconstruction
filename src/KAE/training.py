# ==================================================
#          IMPORTS & PATH CONFIGURATION            #
# ==================================================
import os, sys, math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import wraps
import torch.optim as optim
from tqdm import tqdm

from .networks import ModReLU


# ==================================================
#                      TRAINER                     #
# ==================================================

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r  took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


class Trainer:
    """
    Full-batch gradient descent trainer for complex-valued Koopman autoencoder.

    Loss functions
    --------------
    'mse'      — standard complex MSE  (use when input is normalised)
    'weighted' — separate weights for missing vs acquired lines

    Optimiser note
    --------------
    separate_lrs - useful if you want to see if using different learning rate for the weights and bias of ModReLU.
    """
    def __init__(self, model, momentum_rate, optimizer_type, separate_lrs,
                 epoch, lr_weight=1e-3, lr_bias=1e-5, device='cuda', l2_param_name='latent',
                 val_split=0.2, early_stopping_patience=50, early_stopping=False):
        self.device         = device
        self.model          = model.to(self.device)
        self.momentum_rate  = momentum_rate
        self.epoch          = epoch
        self.optimizer_type = optimizer_type
        self.separate_lrs   = bool(separate_lrs)   # unwrap [False] → False
        self.lr_weight      = lr_weight
        self.lr_bias        = lr_bias
        self.mse_loss       = nn.MSELoss(reduction='sum').to(self.device)
        self.val_split      = val_split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = early_stopping


        # Uniquement pour outer encoder et decoder - class KR
        # self.outer_weights = [
        #     p for name, p in self.model.named_parameters() 
        #     if ('outer_encoder' in name or 'outer_decoder' in name) and 'weight' in name
        # ]
        
        self.params_w = []

        # .lower()
        l2_param_name_lower = l2_param_name.lower() if l2_param_name else ""

        if l2_param_name_lower == 'latent':
            self.params_w = [
                p for name, p in self.model.named_parameters() 
                if ('latent' in name.lower()) and 'weight' in name
            ]
        elif l2_param_name_lower == 'encoderndecoder':
            self.params_w = [
                p for name, p in self.model.named_parameters() 
                if ('encoder' in name.lower() or 'decoder' in name.lower()) and 'weight' in name
            ]
        elif l2_param_name_lower == 'encoder':
            self.params_w = [
                p for name, p in self.model.named_parameters() 
                if ('encoder' in name.lower()) and 'weight' in name
            ]
        elif l2_param_name_lower == 'decoder': 
            self.params_w = [
                p for name, p in self.model.named_parameters() 
                if ('decoder' in name.lower()) and 'weight' in name
            ]
        elif l2_param_name_lower == 'all':
            self.params_w = [
                p for name, p in self.model.named_parameters() 
                if ('encoder' in name.lower() or 'decoder' in name.lower() or 'latent' in name.lower()) and 'weight' in name
            ]

    def _configure_optimizer(self):
        if self.separate_lrs:
            self.params_w, self.params_b, self.params_act = [], [], []
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    self.params_w.append(m.weight)
                    if m.bias is not None:
                        self.params_b.append(m.bias)
                elif isinstance(m, (ModReLU)):
                    self.params_act.extend(list(m.parameters()))

            groups = [
                {'params': self.params_w,   'lr': self.lr_weight},
                {'params': self.params_b,   'lr': self.lr_weight},
                # Activation thresholds: slower lr — they tune fine-grained
                # sparsity/thresholding, not the main signal path
                {'params': self.params_act, 'lr': self.lr_bias},
            ]
        else:
            groups = [{'params': self.model.parameters(), 'lr': self.lr_weight}]

        _opt = {
            'sgd':     lambda g: torch.optim.SGD(g, momentum=self.momentum_rate or 0.9),
            'adam':    lambda g: torch.optim.Adam(g),
            'adam_l2': lambda g: torch.optim.Adam(g, weight_decay=1e-5),
            'adamw':   lambda g: torch.optim.AdamW(g, weight_decay=1e-1),
            'rmsprop': lambda g: torch.optim.RMSprop(g, alpha=0.99),
        }
        if self.optimizer_type not in _opt:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type!r}")
        self.optimizer = _opt[self.optimizer_type](groups)

    def _loss_mse(self, pred, tgt):
        return (self.mse_loss(pred.real, tgt.real)
                + self.mse_loss(pred.imag, tgt.imag))


    def _loss_weighted(self, pred, tgt, mask_src, mask_tgt, lambda_t=1.0, lambda_s=1.0):
        """
        alpha: weight on MISSING lines (primary reconstruction objective)
        beta:  weight on ACQUIRED lines (data consistency)
        Setting beta > 0 prevents the network from ignoring acquired lines.
        """
        sq    = torch.abs(pred - tgt) ** 2
        l_tgt = torch.sum(sq[mask_tgt]) if mask_tgt.any() else sq.new_zeros(())
        l_src = torch.sum(sq[mask_src]) if mask_src.any() else sq.new_zeros(())
        return lambda_t * l_tgt + lambda_s * l_src



    @timing
    def train(self, inputs, outputs,
              loss_type='mse', lambda_t=1.0, lambda_s=0.0,
              l2=False, l2_lambda=1e-4, silent=False):
        """
        Full-batch gradient descent training.

        All N ACS patches are processed in a SINGLE forward/backward call
        per epoch — no mini-batching.  This is intentional:
          - ACS size is small (~7 000–15 000 patches) → fits in GPU RAM.
          - Full-batch has deterministic, stable gradient direction.
          - Avoids noisy gradient estimates from tiny ACS mini-batches.
        """
        self._configure_optimizer()

        N     = inputs.shape[0]
        inp_t = torch.tensor(inputs.reshape(N, -1),
                             dtype=torch.complex64, device=self.device)
        out_t = torch.tensor(outputs.reshape(N, -1),
                             dtype=torch.complex64, device=self.device)

        mask_tgt = (torch.abs(inp_t) == 0)
        mask_src = ~mask_tgt

        losses       = []
        log_interval = max(1, self.epoch // 10)
        print(loss_type, "lambdas s - t : ", lambda_t, lambda_s)
        for ep in tqdm(range(self.epoch), desc="Training", leave=False, disable=silent):
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(inp_t).squeeze()

            if loss_type == 'weighted':
                loss = self._loss_weighted(pred, out_t, mask_src, mask_tgt,
                                           lambda_t=lambda_t, lambda_s=lambda_s)
            else:
                loss = self._loss_mse(pred, out_t)

            if l2:                
                l2_penalty = 0
                l2_penalty += l2_lambda * sum(p.abs().pow(2).sum() for p in self.params_w)
                loss = loss + l2_penalty

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if ep % log_interval == 0:
                print(f"    ep {ep:5d}/{self.epoch}  loss={loss.item():.4e}")

        return {"losses": losses}




