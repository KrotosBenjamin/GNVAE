"""
Module containing the encoders.
"""
import torch
import numpy as np
from torch import nn

# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval(f"Encoder{model_type}")


class EncoderBurgess(nn.Module):
    def __init__(self, img_size, latent_dim=10, hidden_dims=None):
        r"""Encoder of the model proposed in [1]. Works for 2D images only.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)
0
        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class EncoderFullyconnected(nn.Module):
    """
    A flexible, fully-connected encoder that can be configured with a variable
    number of layers and neurons.

    Parameters
    ----------
    img_size : tuple of ints
        Size of the input. E.g., (1, 28, 28) or (1, 784).
    latent_dim : int, default: 128
        Dimensionality of the latent space.
    hidden_dims : list of ints, optional
        A list where each integer is the number of neurons in a hidden
        layer. The number of hidden layers is determined by the length of
        this list. If None, a default architecture is used.

    Example
    -------
    # To replicate EncoderFullyconnected1:
    encoder1 = EncoderFullyconnected(img_size, latent_dim, hidden_dims=[128, 64, 32])

    # To replicate EncoderFullyconnected2:
    encoder2 = EncoderFullyconnected(img_size, latent_dim, hidden_dims=[4096, 1024])

    # To replicate EncoderFullyconnected3:
    encoder3 = EncoderFullyconnected(img_size, latent_dim, hidden_dims=[1024, 1024])

    # To replicate EncoderFullyconnected4:
    encoder4 = EncoderFullyconnected(img_size, latent_dim, hidden_dims=[128, 32])

    # To replicate EncoderFullyconnected5:
    encoder5 = EncoderFullyconnected(img_size, latent_dim, hidden_dims=[128])
    """
    def __init__(self, img_size, latent_dim=128, hidden_dims=None):
        super(EncoderFullyconnected, self).__init__()

        # Set a default architecture if none is provided
        if hidden_dims is None:
            hidden_dims = [128, 64, 32] # Default mirrors original Encoder 1

        self.latent_dim = latent_dim
        self.img_size = img_size

        # Calculate the input dimension by flattening the input shape
        input_dim = int(np.product(img_size))

        # Dynamically build the encoder layers
        modules = []
        all_dims = [input_dim] + hidden_dims
        for i in range(len(all_dims) - 1):
            modules.append(nn.Linear(all_dims[i], all_dims[i+1]))
            modules.append(nn.ReLU())

        # self.encoder holds all the hidden layers
        self.encoder = nn.Sequential(*modules)

        # Final layers to produce mean and log-variance
        self.mu_logvar_gen = nn.Linear(hidden_dims[-1], self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Flatten the input and pass through the encoder
        x = x.view(batch_size, -1)
        x = self.encoder(x)

        # Generate mu and logvar
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class EncoderFullyconnected5(nn.Module):
    """ Fully connected encoder 5

         self.latent_dim = latent_dim
        self.img_size = img_size

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
        - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

    """
    def __init__(self, img_size, latent_dim=128, hidden_dims=None):

        super(EncoderFullyconnected5, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        dims = [128]

        self.lin1 = nn.Linear(np.product(img_size), dims[0])
        self.mu_logvar_gen = nn.Linear(dims[0], self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar
