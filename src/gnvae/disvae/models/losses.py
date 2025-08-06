"""
Module containing all vae losses.
"""
import abc
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from .discriminator import Discriminator
from gnvae.disvae.utils.math import (
    log_density_gaussian,
    matrix_log_density_gaussian,
    log_importance_weight_matrix
)

LOSSES = ["VAE", "betaH", "betaB", "factor", "btcvae"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]

def get_loss_f(loss_name: str, **kwargs_parse) -> nn.Module:
    """
    Return the correct loss function given the argparse arguments.

    Parameters
    ----------
    loss_name : str
        Loss type name.
    kwargs_parse : dict
        Keyword arguments dictionary containing parameters like 'rec_dist',
        'reg_anneal', 'device', 'latent_dim', 'lr_disc', 'n_data', etc.

    Returns
    -------
    loss_function : BaseLoss subclass instance
    """
    kwargs_all = dict(rec_dist=kwargs_parse.get("rec_dist", "bernoulli"),
                      steps_anneal=kwargs_parse.get("reg_anneal", 0))
    if loss_name == "betaH":
        return BetaHLoss(beta=kwargs_parse.get("betaH_B", 4), **kwargs_all)
    elif loss_name == "VAE":
        return BetaHLoss(beta=1, **kwargs_all)
    elif loss_name == "betaB":
        return BetaBLoss(C_init=kwargs_parse.get("betaB_initC", 0.0),
                         C_fin=kwargs_parse.get("betaB_finC", 20.0),
                         gamma=kwargs_parse.get("betaB_G", 100.0),
                         **kwargs_all)
    elif loss_name == "factor":
        # Defensive default values
        device = kwargs_parse.get("device", torch.device("cpu"))
        gamma = kwargs_parse.get("factor_G", 10.0)
        latent_dim = kwargs_parse.get("latent_dim")
        lr_disc = kwargs_parse.get("lr_disc", 5e-5)
        if latent_dim is None:
            raise ValueError("Parameter 'latent_dim' must be provided for 'factor' loss.")
        return FactorKLoss(device=device,
                           gamma=gamma,
                           disc_kwargs=dict(latent_dim=latent_dim),
                           optim_kwargs=dict(lr=lr_disc, betas=(0.5, 0.9)),
                           **kwargs_all)
    elif loss_name == "btcvae":
        n_data = kwargs_parse.get("n_data")
        if n_data is None:
            raise ValueError("Parameter 'n_data' must be provided for 'btcvae' loss.")
        return BtcvaeLoss(n_data=n_data,
                          alpha=kwargs_parse.get("btcvae_A", 1.0),
                          beta=kwargs_parse.get("btcvae_B", 6.0),
                          gamma=kwargs_parse.get("btcvae_G", 1.0),
                          **kwargs_all)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every : int, optional
        Every how many steps to record the loss. (default: 50)
    rec_dist : {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction likelihood distribution on each pixel.
    steps_anneal : int, optional
        Number of annealing steps to gradually add the regularization.
    """
    def __init__(self, record_loss_every: int = 50, rec_dist: str = "bernoulli",
                 steps_anneal: int = 0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        if rec_dist not in RECON_DIST:
            raise ValueError(f"Unknown reconstruction distribution: {rec_dist}")
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self,
                 data: torch.Tensor,
                 recon_data: torch.Tensor,
                 latent_dist: Tuple[torch.Tensor, torch.Tensor],
                 is_train: bool,
                 storer: Optional[Dict[str, list]],
                 **kwargs) -> torch.Tensor:
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data, shape (batch_size, channels, height, width).
        recon_data : torch.Tensor
            Reconstructed data, same shape as input.
        latent_dist : tuple of torch.Tensor
            Sufficient statistics e.g. (mean, logvar), each of shape (batch_size, latent_dim).
        is_train : bool
            Whether currently in training mode.
        storer : dict or None
            Dictionary to log intermediate values for visualization.

        Returns
        -------
        loss : torch.Tensor
            Computed loss scalar tensor.
        """
        pass

    def _pre_call(self, is_train: bool, storer: Optional[Dict[str, list]]) -> Optional[Dict[str, list]]:
        if is_train:
            self.n_train_steps += 1
        # Record loss every `record_loss_every` steps, including step 1
        # for logging start
        if not is_train or (self.n_train_steps % self.record_loss_every == 1):
            return storer
        else:
            return None


class BetaHLoss(BaseLoss):
    """
    Beta-VAE loss [1].

    Parameters
    ----------
    beta : float
        Weight of KL divergence term (default 4).
    """
    def __init__(self, beta: float = 4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train,
                 storer=None, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer=storer)
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)
        loss = rec_loss + anneal_reg * (self.beta * kl_loss)

        if storer is not None:
            storer.setdefault('loss', []).append(loss.item())

        return loss


class BetaBLoss(BaseLoss):
    """
    Beta-B VAE loss [1].

    Parameters
    ----------
    C_init : float
        Starting annealed capacity C.
    C_fin : float
        Final annealed capacity.
    gamma : float
        Weight of KL divergence term.
    """
    def __init__(self, C_init: float = 0., C_fin: float = 20.,
                 gamma: float = 100., **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, data, recon_data, latent_dist, is_train,
                 storer=None, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer=storer)

        C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.steps_anneal)
             if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        if storer is not None:
            storer.setdefault('loss', []).append(loss.item())

        return loss


class FactorKLoss(BaseLoss):
    """
    Factor VAE loss [1].

    Parameters
    ----------
    device : torch.device
        Device to run the loss computations.
    gamma : float
        Weight of the total correlation (TC) term.
    disc_kwargs : dict
        Keyword arguments for the discriminator network.
    optim_kwargs : dict
        Keyword arguments for the discriminator optimizer.
    anneal_discriminator : bool, optional
        Whether to anneal the discriminator loss to stabilize training.
        Default False.
    """
    def __init__(self, device: torch.device,
                 gamma: float = 10.,
                 disc_kwargs: Optional[dict] = None,
                 optim_kwargs: Optional[dict] = None,
                 anneal_discriminator: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        disc_kwargs = disc_kwargs or {}
        optim_kwargs = optim_kwargs or dict(lr=5e-5, betas=(0.5, 0.9))
        self.gamma = gamma
        self.device = device
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)
        self.anneal_discriminator = anneal_discriminator

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Use `call_optimize` method for FactorKLoss to train discriminator.")

    def call_optimize(self,
                      data: torch.Tensor,
                      model: nn.Module,
                      optimizer: optim.Optimizer,
                      storer: Optional[Dict[str, list]] = None) -> torch.Tensor:
        storer = self._pre_call(model.training, storer)

        batch_size = data.size(0)
        if batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even for FactorKLoss, got {batch_size}.")

        half_batch_size = batch_size // 2
        data1, data2 = data.split(half_batch_size)

        # Forward pass for first half batch
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch, storer=storer, distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer=storer)

        d_z = self.discriminator(latent_sample1)
        # TC loss is difference between discriminator logits for true vs permuted codes
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)

        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        if storer is not None:
            storer.setdefault('loss', []).append(vae_loss.item())
            storer.setdefault('tc_loss', []).append(tc_loss.item())

        if not model.training:
            return vae_loss

        # VAE optimizer
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)

        # Discriminator training
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        zeros = torch.zeros(half_batch_size, dtype=torch.long, device=self.device)
        ones = torch.ones_like(zeros)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))

        if self.anneal_discriminator:
            d_tc_loss = anneal_reg * d_tc_loss

        self.optimizer_d.zero_grad()
        d_tc_loss.backward()

        optimizer.step()
        self.optimizer_d.step()

        if storer is not None:
            storer.setdefault('discrim_loss', []).append(d_tc_loss.item())

        return vae_loss


class BtcvaeLoss(BaseLoss):
    """
    Beta-TCVAE loss [1].

    Parameters
    ----------
    n_data : int
        Number of data points in training set.
    alpha : float
        Weight of the Mutual Information term.
    beta : float
        Weight of the Total Correlation term.
    gamma : float
        Weight of the dimension-wise KL term.
    is_mss : bool, optional
        Whether to use minibatch stratified sampling (MSS) or weighted sampling.
    """
    def __init__(self,
                 n_data: int,
                 alpha: float = 1.0,
                 beta: float = 6.0,
                 gamma: float = 1.0,
                 is_mss: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.is_mss = is_mss

    def __call__(self,
                 data: torch.Tensor,
                 recon_batch: torch.Tensor,
                 latent_dist: Tuple[torch.Tensor, torch.Tensor],
                 is_train: bool,
                 storer: Optional[Dict[str, list]] = None,
                 latent_sample: Optional[torch.Tensor] = None) -> torch.Tensor:
        storer = self._pre_call(is_train, storer)
        if latent_sample is None:
            raise ValueError("latent_sample must be provided for BtcvaeLoss")

        batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch, storer=storer, distribution=self.rec_dist)

        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
            latent_sample,
            latent_dist,
            self.n_data,
            is_mss=self.is_mss
        )

        # Mutual Information term: I[z;x] = E_x KL[q(z|x)||q(z)]
        mi_loss = (log_q_zCx - log_qz).mean()

        # Total Correlation term: TC[z] = KL[q(z)||prod_i q(z_i)]
        tc_loss = (log_qz - log_prod_qzi).mean()

        # Dimension-wise KL: KL[q(z) || p(z)]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)

        loss = rec_loss + (self.alpha * mi_loss +
                           self.beta * tc_loss +
                           anneal_reg * self.gamma * dw_kl_loss)

        if storer is not None:
            storer.setdefault('loss', []).append(loss.item())
            storer.setdefault('mi_loss', []).append(mi_loss.item())
            storer.setdefault('tc_loss', []).append(tc_loss.item())
            storer.setdefault('dw_kl_loss', []).append(dw_kl_loss.item())

            # Compute KL for comparison/storage
            _kl_normal_loss(*latent_dist, storer=storer)

        return loss


# Helper functions
def _reconstruction_loss(data: torch.Tensor, recon_data: torch.Tensor,
                         distribution: str = "bernoulli",
                         storer: Optional[Dict[str, list]] = None) -> torch.Tensor:
    """
    Calculate reconstruction loss (negative log likelihood) for a batch.

    Parameters
    ----------
    data : torch.Tensor
        Target data, shape (batch, channels, height, width).
    recon_data : torch.Tensor
        Reconstructed data, same shape as data.
    distribution : str
        One of {"bernoulli", "gaussian", "laplace"}.
    storer : dict or None
        Store loss values if provided.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss tensor.
    """
    batch_size, n_chan, height, width = recon_data.size()

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * 3  # empirical scaling to match bernoulli levels
        # Mask zero-loss (to avoid NaN?): considered unnecessary unless
        # observed - can add epsilon if needed
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    loss = loss / batch_size

    if storer is not None:
        storer.setdefault('recon_loss', []).append(loss.item())

    return loss


def _kl_normal_loss(mean: torch.Tensor, logvar: torch.Tensor,
                   storer: Optional[Dict[str, list]] = None) -> torch.Tensor:
    """
    KL divergence between diagonal Gaussian q(z|x) and prior p(z) = N(0,I).

    Parameters
    ----------
    mean : torch.Tensor
        Mean, shape (batch_size, latent_dim)
    logvar : torch.Tensor
        Log variance, shape (batch_size, latent_dim)
    storer : dict or None
        Store kl loss values if provided.

    Returns
    -------
    total_kl : torch.Tensor
        Scalar KL divergence loss tensor.
    """
    latent_dim = mean.size(1)
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer.setdefault('kl_loss', []).append(total_kl.item())
        for i in range(latent_dim):
            key = f'kl_loss_{i}'
            storer.setdefault(key, []).append(latent_kl[i].item())

    return total_kl


def _permute_dims(latent_sample: torch.Tensor) -> torch.Tensor:
    """
    Permute samples across the batch for each latent dimension
    (Algorithm 1 from [1]).

    Parameters
    ----------
    latent_sample : torch.Tensor
        Latent samples, shape (batch_size, latent_dim)

    Returns
    -------
    permuted : torch.Tensor
        Permuted tensor, same shape as input.

    References
    ----------
    [1] Kim and Mnih, "Disentangling by factorising", 2018.
    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()
    for z in range(dim_z):
        pi = torch.randperm(batch_size, device=latent_sample.device)
        perm[:, z] = latent_sample[pi, z]
    return perm


def linear_annealing(init: float, fin: float, step: int,
                     annealing_steps: int) -> float:
    """
    Linear annealing schedule from init to fin over annealing_steps.

    Returns fin if annealing_steps is 0.

    Parameters
    ----------
    init : float
        Start value.
    fin : float
        Final value.
    step : int
        Current step.
    annealing_steps : int
        Total number of annealing steps.

    Returns
    -------
    float
        Current annealed value.
    """
    if annealing_steps == 0:
        return fin
    if fin <= init:
        raise ValueError(f"fin must be greater than init (got init={init}, fin={fin})")
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


def _get_log_pz_qz_prodzi_qzCx(
    latent_sample: torch.Tensor,
    latent_dist: Tuple[torch.Tensor, torch.Tensor],
    n_data: int,
    is_mss: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute log densities used for Beta-TCVAE loss.

    Parameters
    ----------
    latent_sample : torch.Tensor
        Latent samples, shape (batch, latent_dim).
    latent_dist : tuple of torch.Tensor
        (mean, logvar) of q(z|x), shape each (batch, latent_dim).
    n_data : int
        Number of data samples in full dataset.
    is_mss : bool
        Whether to use minibatch stratified sampling.

    Returns
    -------
    log_pz, log_qz, log_prod_qzi, log_q_zCx : torch.Tensor
        Per-sample log densities.
    """
    batch_size, hidden_dim = latent_sample.shape

    # log q(z|x) per dimension, summed over latent dims
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # log p(z) with zero mean and var = 1
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(dim=1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx
