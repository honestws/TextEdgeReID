"""
---
title: Generate images using stable diffusion with a prompt
summary: >
 Generate images using stable diffusion with a prompt
---

# Generate images using [stable diffusion](../index.html) with a prompt
"""

import torch
from latent.ddim import DDIMSampler
from latent.ddpm import DDPMSampler
from latent.latent_diffusion import LatentDiffusion
from latent.util import load_model


class LatentDiffusionModel:
    """
    ### Text to image class
    """
    model: LatentDiffusion

    def __init__(self,
                 clip_model,
                 vae,
                 dataloader,
                 lr: float,
                 sampler_name: str,
                 n_steps: int = 50,
                 ddim_eta: float = 0.0,
                 ):
        """
        :param clip_model: clip model
        :param vae: autoencoder
        :param lr: learning rate
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        # Create dataloader
        self.data_loader = dataloader
        # Load [latent diffusion model](../latent_diffusion.html)
        self.model = load_model(clip_model, vae)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Get device
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # Move the model to device
        self.model.to(self.device)

        # Initialize [sampler](../sampler/index.html)
        if sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model,
                                       n_steps=n_steps,
                                       ddim_eta=ddim_eta)
        elif sampler_name == 'ddpm':
            self.sampler = DDPMSampler(self.model)

    def train(self):
        """
        ### Train
        """

    @torch.no_grad()
    def infer(self,
              dest_path: str,
              prompt: str,
              batch_size: int = 3,
              h: int = 512, w: int = 512,
              uncond_scale: float = 7.5
              ):
        """
        :param dest_path: is the path to store the generated images
        :param batch_size: is the number of images to generate in a batch
        :param prompt: is the prompt to generate images with
        :param h: is the height of the image
        :param w: is the width of the image
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        """
        # Number of channels in the image
        c = 4
        # Image to latent space resolution reduction
        f = 8

        # Make a batch of prompts
        prompts = batch_size * [prompt]

        # AMP auto casting
        with torch.cuda.amp.autocast():
            # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
            if uncond_scale != 1.0:
                un_cond = self.model.get_text_conditioning(batch_size * [""])
            else:
                un_cond = None
            # Get the prompt embeddings
            cond = self.model.get_text_conditioning(prompts)
            # [Sample in the latent space](../sampler/index.html).
            # `x` will be of shape `[batch_size, c, h / f, w / f]`
            x = self.sampler.sample(cond=cond,
                                    shape=[batch_size, c, h // f, w // f],
                                    uncond_scale=uncond_scale,
                                    uncond_cond=un_cond)
            # Decode the image from the [autoencoder](../model/autoencoder.html)
            # images = self.model.autoencoder_decode(x)

        # Save images
        # save_images(images, dest_path, 'txt_')

