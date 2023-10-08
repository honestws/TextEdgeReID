import torch
import torch.nn.functional as F
from torch import nn
from vae.base import BaseVAE
from typing import TypeVar, List
Tensor = TypeVar('torch.tensor')


class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: List,
                 hidden_dims: list = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 64, 32, 4]

        # Build Encoder
        for i, h_dim in enumerate(hidden_dims):
            if (i+1) % 3 == 0 and h_dim == 128:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=(2, 2), padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
            elif (i+1) % 3 == 0:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=(2, 1), padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=(1, 1), padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(
                        nn.Conv2d(h_dim, out_channels=h_dim,
                                  kernel_size=3, stride=(1, 1), padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
        self.fc_var = nn.Sequential(
                        nn.Conv2d(h_dim, out_channels=h_dim,
                                  kernel_size=3, stride=(1, 1), padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())


        # Build Decoder
        modules = []

        self.decoder_input = nn.Sequential(
                        nn.Conv2d(h_dim, out_channels=h_dim,
                                  kernel_size=3, stride=(1, 1), padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == 128:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i], out_channels=hidden_dims[i + 1],
                                  kernel_size=3, stride=(2, 2), padding=1, output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )
            elif hidden_dims[i] == 4:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i], out_channels=hidden_dims[i + 1],
                                  kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i], out_channels=hidden_dims[i + 1],
                                  kernel_size=3, stride=(1, 1), padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=(1, 1),
                                               padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        feat = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(feat)
        log_var = self.fc_var(feat)

        return [feat, mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        lat = self.decoder_input(z)
        r = self.decoder(lat)
        r = self.final_layer(r)
        return lat, r

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        feat, mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z)[1], input, feat, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[3]
        log_var = args[4]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input, reduction='sum')


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = (1, 2, 3)), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self, num_samples:int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples, *self.latent_dim)

        z = z.to("cuda:0")

        samples = self.decode(z)[0]
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]