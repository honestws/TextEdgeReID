"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# Utility functions for [stable diffusion](index.html)
"""

import random
import numpy as np
import torch
from latent.latent_diffusion import LatentDiffusion
from latent.unet import UNetModel


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def set_seed(seed: int):
    """
    ### Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(clip, vae, latent_scaling_factor):
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """

    # Initialize the autoencoder
    clip_checkpoint = torch.load('checkpoints/clip.pt')
    vae_checkpoint = torch.load('checkpoints/vae.pt')
    clip.load_state_dict(clip_checkpoint['net'])
    vae.load_state_dict(vae_checkpoint['net'])
    eps_model = UNetModel(in_channels=4,
                          out_channels=4,
                          channels=320,
                          attention_levels=[0, 1, 2],
                          n_res_blocks=2,
                          channel_multipliers=[1, 2, 4, 4],
                          n_heads=8)
    model = LatentDiffusion(linear_start=0.00085,
                            linear_end=0.0120,
                            n_steps=1000,
                            latent_scaling_factor=latent_scaling_factor,
                            autoencoder=vae,
                            clip_embedder=clip,
                            unet_model=eps_model)

    #
    model.eval()
    return model
