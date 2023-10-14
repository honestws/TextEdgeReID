"""
---
title: Latent Diffusion Models
summary: >
 Annotated PyTorch implementation/tutorial of latent diffusion models from paper
 High-Resolution Image Synthesis with Latent Diffusion Models
---

# Latent Diffusion Models

Latent diffusion models use an auto-encoder to map between image space and
latent space. The diffusion model works on the latent space, which makes it
a lot easier to train.
It is based on paper
[High-Resolution Image Synthesis with Latent Diffusion Models](https://papers.labml.ai/paper/2112.10752).

They use a pre-trained auto-encoder and train the diffusion U-Net on the latent
space of the pre-trained auto-encoder.

For a simpler diffusion implementation refer to our [DDPM implementation](../ddpm/index.html).
We use same notations for $\alpha_t$, $\beta_t$ schedules, etc.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import clip
from latent.utils import gather


class DiffusionWrapper(nn.Module):
    """
    *This is an empty wrapper class around the [U-Net](model/unet.html).
    We keep this to have the same model structure as
    [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
    so that we do not have to map the checkpoint weights explicitly*.
    """

    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, time_steps, context)


class LatentDiffusion(nn.Module):
    """
    ## Latent diffusion model

    This contains following components:

    * [AutoEncoder](model/autoencoder.html)
    * [U-Net](model/unet.html) with [attention](model/unet_attention.html)
    * [CLIP embeddings generator](model/clip_embedder.html)
    """

    def __init__(self,
                 unet_model,
                 vae_decoder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                 device: str
                 ):
        """
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param vae_decoder: is the decoder from [AutoEncoder]
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()
        # Wrap the [U-Net](model/unet.html) to keep the same model structure as
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
        self.eps_model = DiffusionWrapper(unet_model)
        # Auto-encoder and scaling factor
        self.vae_decoder = vae_decoder
        self.latent_scaling_factor = latent_scaling_factor
        # [CLIP transformer](model/clip_embedder.html)

        self.device_ = device

        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # $\alpha_t = 1 - \beta_t$
        alpha = 1. - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.eps_model.parameters())).device

    def get_text_conditioning(self, prompts: List[str]):
        """
        ### Get [CLIP embeddings](model/clip_embedder.html) for a list of text prompts
        """
        txts = clip.tokenize(prompts, truncate=True).to(self.device_)
        return self.text_encoder(txts)[1]

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.eps_model(x, t, context)

    def loss(self, x0: torch.Tensor, embeddings: torch.Tensor,
             uncond_embedding: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        #### Compute Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        batch_size = embeddings.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Unconditional embedding
        un_cond = uncond_embedding.repeat(batch_size, 1)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.get_eps(xt, t, embeddings,
                                   uncond_scale=self.latent_scaling_factor,
                                   uncond_cond=un_cond)

        # MSE loss
        return F.mse_loss(noise, eps_theta)

    def get_eps(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, *,
                uncond_scale: float, uncond_cond: Optional[torch.Tensor]):
        """
        ## Get $\epsilon(x_t, c)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param t: is $t$ of shape `[batch_size]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # When the scale $s = 1$
        # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$
        if uncond_cond is None or uncond_scale == 1.:
            return self.eps_model(x, t, c)

        # Duplicate $x_t$ and $t$
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        # Concatenated $c$ and $c_u$
        c_in = torch.cat([uncond_cond, c])
        # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
        e_t_uncond, e_t_cond = self.eps_model(x_in, t_in, c_in).chunk(2)
        # Calculate
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

        #
        return e_t

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

