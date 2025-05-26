import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy
from modules.criterions import SeqKD
from utils.Glossloss import *


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

class GaussianDiffusion(nn.Module):
    def __init__(
        self, model, betas=[], loss_type="l2", ema_decay=0.9999, ema_start=2000, ema_update_rate=1,
    ):
        super().__init__()
        self.model      = model
        self.ema_model  = deepcopy(model)

        self.ema                = EMA(ema_decay)
        self.ema_decay          = ema_decay
        self.ema_start          = ema_start
        self.ema_update_rate    = ema_update_rate
        self.step               = 0

        # l1或者l2损失
        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type      = loss_type
        self.num_timesteps  = len(betas)

        alphas              = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas)

        # 转换成torch.tensor来处理
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

        self.CTC_loss = torch.nn.CTCLoss(reduction='none', zero_infinity=False)

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y, v_len, label_len, use_ema=True):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y, v_len, label_len)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y, v_len, label_len)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, x, y, v_len, label_len, g, use_ema=False):
        alpha = 1e-4
        b, _, _ = x.shape
        weights = generate_linear_schedule(self.num_timesteps, 0.01, 0.0001)
        loss = 0
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t]).repeat(b).cuda()
            x = self.remove_noise(x, t_batch, y, v_len, label_len, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            loss += self.CTC_loss(x, y, label_len, v_len) * weights[t] + Glossloss()(x, y) * weights[t]  # 计算RSM与GFE
        sample_output = alpha * x
        # Calculate the total loss
        loss = loss / self.num_timesteps
        return sample_output, loss

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t,  x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x, t, y, v_len, label_len):
        # x, noise [batch_size, 3, 64, 64]
        noise           = torch.randn_like(x)

        perturbed_x     = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, y, v_len, label_len)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)
        return loss

    def forward(self, x, y, v_len, label_len):
        b, l, d = x.shape
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, y, v_len, label_len)

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)