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
from tqdm import tqdm

from latent.unet import UNetModel


class LatentDiffusionModel:
    """
    ### Text to image class
    """
    model: LatentDiffusion

    def __init__(self,
                 clip_model,
                 vae,
                 uncond_scale: float,
                 device: str,
                 sampler_name: str,
                 n_steps: int = 50,
                 ddim_eta: float = 0.0
                 ):
        """
        :param clip_model: clip model
        :param vae: autoencoder
        :param device: CUDA device
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        # Load [latent diffusion model](../latent_diffusion.html)
        self.model = load_model(clip_model, vae, uncond_scale, device)
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
        # Iterate through the dataset
        num_batch = len(self.data_loader)
        pbar = tqdm(enumerate(self.data_loader), total=num_batch)
        for n_iter, (imgs, pids, captions) in pbar:
            self.optimizer.zero_grad()
            # Generate data by VAE
            data = self.model.first_stage_model.sample(len(captions))
            # Calculate loss
            loss = self.model.loss(data, captions)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()

    @torch.no_grad()
    def infer(self,
              prompt: str,
              batch_size: int = 3,
              h: int = 512, w: int = 512,
              uncond_scale: float = 7.5
              ):
        """
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



def load_model(clip, vae, scale, device):
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
                            latent_scaling_factor=scale,
                            autoencoder=vae,
                            clip_embedder=clip,
                            unet_model=eps_model,
                            device=device)

    #
    model.eval()
    return model