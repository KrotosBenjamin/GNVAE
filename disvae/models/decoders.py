"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

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

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x


class DecoderFullyconnected1(nn.Module):
    """ Fully connected decoder 1
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
        - 3 fully connected layers (each of 256 units)
        - Latent distribution:
        - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

     """

    def __init__(self, img_size, latent_dim=10):
        super(DecoderFullyconnected1, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size

        dims = [128, 64, 32]
        
        self.lin0 = nn.Linear(latent_dim, dims[2])
        self.lin1 = nn.Linear(dims[2], dims[1])
        self.lin2 = nn.Linear(dims[1], dims[0])
        self.lin3 = nn.Linear(dims[0], np.product(self.img_size))
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin0(z))
        x = torch.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        x = x.view(batch_size, *self.img_size)

        return x

    

class DecoderFullyconnected2(nn.Module):
    """ Fully connected decoder 2
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
        - 3 fully connected layers (each of 256 units)
        - Latent distribution:
        - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

     """

    def __init__(self, img_size, latent_dim=10):
        super(DecoderFullyconnected2, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size

        dims = [4096, 1024]
        
        self.lin0 = nn.Linear(latent_dim, dims[1])
        self.lin1 = nn.Linear(dims[1], dims[0])
        self.lin2 = nn.Linear(dims[0], np.product(self.img_size))
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin0(z))
        x = torch.relu(self.lin1(x))
        #x = torch.sigmoid(self.lin2(x)) # dropping the sigmoid
        x = self.lin2(x)
        x = x.view(batch_size, *self.img_size)

        return x

    
class DecoderFullyconnected3(nn.Module):
    """ Fully connected decoder 3
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
        - 3 fully connected layers (each of 256 units)
        - Latent distribution:
        - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

     """

    def __init__(self, img_size, latent_dim=10):
        super(DecoderFullyconnected3, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size

        dims = [1024, 1024]
        
        self.lin0 = nn.Linear(latent_dim, dims[1])
        self.lin1 = nn.Linear(dims[1], dims[0])
        self.lin2 = nn.Linear(dims[0], np.product(self.img_size))
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin0(z))
        x = torch.relu(self.lin1(x))
        #x = torch.sigmoid(self.lin2(x)) # dropping the sigmoid
        x = self.lin2(x)
        x = x.view(batch_size, *self.img_size)

        return x

    

class DecoderFullyconnected4(nn.Module):
    """ Fully connected decoder 4
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
        - 3 fully connected layers (each of 256 units)
        - Latent distribution:
        - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

     """

    def __init__(self, img_size, latent_dim=10):
        super(DecoderFullyconnected4, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size

        dims = [128, 32]
        
        self.lin0 = nn.Linear(latent_dim, dims[1])
        self.lin1 = nn.Linear(dims[1], dims[0])
        self.lin2 = nn.Linear(dims[0], np.product(self.img_size))
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin0(z))
        x = torch.relu(self.lin1(x))
        #x = torch.sigmoid(self.lin2(x)) # dropping the sigmoid
        x = self.lin2(x)
        x = x.view(batch_size, *self.img_size)

        return x


class DecoderFullyconnected5(nn.Module):
    """ Fully connected decoder 5
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
        - 3 fully connected layers (each of 256 units)
        - Latent distribution:
        - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

     """

    def __init__(self, img_size, latent_dim=10):
        super(DecoderFullyconnected5, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size

        dims = [128]
        
        self.lin0 = nn.Linear(latent_dim, dims[0])
        self.lin1 = nn.Linear(dims[0], np.product(self.img_size))
        
    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin0(z))
        #x = torch.sigmoid(self.lin2(x)) # dropping the sigmoid
        x = self.lin1(x)
        x = x.view(batch_size, *self.img_size)

        return x
