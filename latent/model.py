"""
---
title: Generate images using stable diffusion with a prompt
summary: >
 Generate images using stable diffusion with a prompt
---

# Generate images using [stable diffusion](../index.html) with a prompt
"""
import torch
from collections import OrderedDict
from latent.ddim import DDIMSampler
from latent.ddpm import DDPMSampler
from latent.latent_diffusion import LatentDiffusion
from latent.unet import UNetModel


class LatentDiffusionModel:
    """
    ### Text to image class
    """
    model: LatentDiffusion

    def __init__(self,
                 vae_decoder,
                 uncond_scale: float,
                 device: str,
                 sampler_name: str,
                 n_steps: int = 50,
                 ddim_eta: float = 0.0
                 ):
        """
        :param vae_decoder: autoencoder
        :param device: CUDA device
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        # Load [latent diffusion model](../latent_diffusion.html)
        self.model= load_model(vae_decoder, uncond_scale, device)
        # Create optimizer
        # Get device
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # Move models to device
        self.model.to(self.device)

        # Initialize [sampler](../sampler/index.html)
        if sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model,
                                       n_steps=n_steps,
                                       ddim_eta=ddim_eta)
        elif sampler_name == 'ddpm':
            self.sampler = DDPMSampler(self.model)

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

def load_model(vae_decoder, scale, device):
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """

    # Initialize the autoencoder
    clip_checkpoint = torch.load('checkpoints/clip.pt')
    clip_transformer_dict = OrderedDict()
    for key, val in clip_checkpoint['net'].items():
        if 'model.transformer.' in key:
            clip_transformer_dict.update({key.split('model.transformer.')[1]: val})
        if 'model.ln_final' in key:
            clip_transformer_dict.update({key.split('model.')[1]: val})
    vae_checkpoint = torch.load('checkpoints/vae.pt')
    vae_dict = OrderedDict()
    for key, val in vae_checkpoint['net'].items():
        if 'decoder_input.' in key:
            vae_dict.update({key.split('decoder_input.')[1]: val})

    vae_decoder.load_state_dict(vae_dict)
    eps_model = UNetModel(in_channels=4,
                          out_channels=4,
                          channels=320,
                          attention_levels=[0, 1, 2],
                          n_res_blocks=2,
                          channel_multipliers=[1, 2, 4, 4],
                          n_heads=8)
    model = LatentDiffusion(unet_model=eps_model, vae_decoder=vae_decoder,
                            latent_scaling_factor=scale, n_steps=1000, linear_start=0.00085, linear_end=0.0120,
                            device=device)

    #
    model.eval()
    return model

