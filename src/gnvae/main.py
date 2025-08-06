#!/usr/bin/env python

import sys
import argparse
import logging
from os.path import join, dirname
from inspect import getsourcefile
from configparser import ConfigParser

from torch import optim

from gnvae.disvae.models.vae import MODELS
from gnvae.utils.visualize import GifTraversalsTraining
from gnvae.disvae import init_specific_model, Trainer, Evaluator
from gnvae.disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from gnvae.utils.datasets import get_dataloaders, get_img_size, DATASETS
from gnvae.disvae.utils.modelIO import save_model, load_model, load_metadata
from .utils.helpers import (
    set_seed,
    get_device,
    get_n_param,
    update_namespace_,
    get_config_section,
    FormatterNoDuplicate,
    create_safe_directory
)

RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

def parse_arguments(cli_args):
    """Parse the command line arguments.

    Parameters
    ----------
    cli_args: list of str
        Arguments to parse (splitted on whitespaces).
    """
    # Temporary parser to extract config file path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', '-c', type=str, default=None,
                            help="Path to the config file (hyperparam.ini).")
    config_args, _ = pre_parser.parse_known_args(cli_args)

    config_path = config_args.config if config_args.config else join(dirname(__file__), "hyperparam.ini")
    default_config = get_config_section([config_path], "Custom")

    description = "PyTorch implementation and evaluation of disentangled VAE and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate,
                                     parents=[pre_parser])

    # General options
    parser.add_argument('name', type=str, help="Name for storing/loading the model.")
    parser.add_argument('-L', '--log-level', default=default_config['log_level'],
                        choices=LOG_LEVELS)
    parser.add_argument('--no-progress-bar', action='store_true',
                        default=default_config['no_progress_bar'],
                        help='Disables progress bar.')
    parser.add_argument('--no-cuda', action='store_true',
                        default=default_config['no_cuda'],
                        help='Disables use of CUDA.')
    parser.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                        help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('--fold', default=None, help='Fold number for caudate dataset.')

    # Training
    parser.add_argument('--checkpoint-every', type=int,
                        default=default_config['checkpoint_every'],
                        help='Save a checkpoint of the trained model every n epoch.')
    parser.add_argument('-d', '--dataset', default=default_config['dataset'],
                        choices=DATASETS, help="Path to training data.")
    parser.add_argument('--gene-expression-filename', default=None,
                        help="CSV file with gene expression values. Genes are rows, samples are columns.")
    parser.add_argument('--gene-expression-dir', default=None,
                        help="Path to 10-fold generated data with X_train.csv and X_test.csv.")
    parser.add_argument('-x', '--experiment', default=default_config['experiment'],
                        choices=EXPERIMENTS,
                        help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    parser.add_argument('-e', '--epochs', type=int, default=default_config['epochs'],
                        help='Maximum number of epochs to run for.')
    parser.add_argument('-b', '--batch-size', type=int, default=default_config['batch_size'],
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=default_config['lr'],
                        help='Learning rate.')

    # Model
    parser.add_argument('-m', '--model-type', default=default_config['model'],
                        choices=MODELS, help='Type of encoder and decoder to use.')
    parser.add_argument('-z', '--latent-dim', type=int, default=default_config['latent_dim'],
                        help='Dimension of the latent variable.')
    parser.add_argument('-l', '--loss', default=default_config['loss'],
                        choices=LOSSES, help="Type of VAE loss function to use.")
    parser.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                        choices=RECON_DIST, help="Form of the likelihood ot use for each pixel.")
    parser.add_argument('-a', '--reg-anneal', type=float, default=default_config['reg_anneal'],
                        help="Number of annealing steps where gradually adding the regularisation.")

    # Loss Specific
    parser.add_argument('--betaH-B', type=float, default=default_config['betaH_B'],
                        help="Weight of the KL.")
    parser.add_argument('--betaB-initC', type=float, default=default_config['betaB_initC'],
                        help="Starting annealed capacity.")
    parser.add_argument('--betaB-finC', type=float, default=default_config['betaB_finC'],
                        help="Final annealed capacity.")
    parser.add_argument('--betaB-G', type=float, default=default_config['betaB_G'],
                        help="Weight of the KL divergence term (gamma in the paper).")
    parser.add_argument('--factor-G', type=float, default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    parser.add_argument('--lr-disc', type=float, default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')
    parser.add_argument('--btcvae-A', type=float, default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    parser.add_argument('--btcvae-G', type=float, default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    parser.add_argument('--btcvae-B', type=float, default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Evaluation
    parser.add_argument('--is-eval-only', action='store_true', default=default_config['is_eval_only'],
                        help='Whether to only evaluate using precomputed model `name`.')
    parser.add_argument('--is-metrics', action='store_true', default=default_config['is_metrics'],
                        help="Whether to compute the disentangled metrcics. For `dsprites` only.")
    parser.add_argument('--no-test', action='store_true', default=default_config['no_test'],
                        help="Whether not to compute the test losses.`")
    parser.add_argument('--eval-batchsize', type=int, default=default_config['eval_batchsize'],
                        help='Batch size for evaluation.')

    args = parser.parse_args(cli_args)

    # Handle experiment presets
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            model_type, dataset = args.experiment.split("_")
            update_namespace_(args, get_config_section([config_path], f"Common_{dataset}"))
            update_namespace_(args, get_config_section([config_path], f"Common_{model_type}"))
        try:
            update_namespace_(args, get_config_section([config_path], args.experiment))
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e

    # Ensure valid gene expression input for geneexpression dataset
    if args.dataset == 'geneexpression':
        if bool(args.gene_expression_filename) == bool(args.gene_expression_dir):
            parser.error("You must specify exactly one of --gene_expression_filename or --gene_expression_dir for gene expression datasets.")

    parser.set_defaults(**vars(args))
    return parser.parse_args(cli_args)


def setup_logging(log_level):
    log_fmt = '%(asctime)s %(levelname)s - %(funcName)s: %(message)s'
    formatter = logging.Formatter(log_fmt, "%H:%M:%S")
    logger = logging.getLogger()
    # Avoid duplicate log output
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level.upper())
    return logger


def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    logger = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info(f"Root directory for saving and loading experiments: {exp_dir}")

    # Training
    if not args.is_eval_only:
        create_safe_directory(exp_dir, logger=logger)

        if args.loss == "factor":
            logger.info("FactorVae: doubling batch and epoch count to simulate 2 batches/iteration consistency.")
            args.batch_size *= 2
            args.epochs *= 2

        extra_kwargs = {'gene_expression_filename': args.gene_expression_filename} if args.gene_expression_filename else {}
        train_loader = get_dataloaders(args.dataset, batch_size=args.batch_size,
                                       logger=logger, **extra_kwargs)
        logger.info(f"Train dataset '{args.dataset}' samples: {len(train_loader.dataset)}")

        # Store image size if available
        args.img_size = getattr(train_loader.dataset, 'img_size', None)
        model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
        logger.info(f'Num parameters in model: {get_n_param(model)}')

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(device)
        loss_f = get_loss_f(args.loss, n_data=len(train_loader.dataset),
                            device=device, **vars(args))
        # Uncomment to enable viz: gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        trainer = Trainer(
            model, optimizer, loss_f,
            device=device, logger=logger, save_dir=exp_dir,
            is_progress_bar=not args.no_progress_bar, gif_visualizer=None
        )
        trainer(train_loader, epochs=args.epochs,
                checkpoint_every=args.checkpoint_every)
        save_model(trainer.model, exp_dir, metadata=vars(args))

    # Evaluation and/or metrics
    if args.is_metrics or not args.no_test:
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        eval_kwargs = {'gene_expression_filename': args.gene_expression_filename} if args.gene_expression_filename else {}
        ## TODO: Need to fix load_metadata so that test data used!
        test_loader = get_dataloaders(
            metadata["dataset"], batch_size=args.eval_batchsize, shuffle=False,
            logger=logger, **eval_kwargs
        )
        loss_f = get_loss_f(args.loss, n_data=len(test_loader.dataset),
                            device=device, **vars(args))
        evaluator = Evaluator(
            model, loss_f, device=device, logger=logger, save_dir=exp_dir,
            is_progress_bar=not args.no_progress_bar
        )
        evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)


def cli():
    args = parse_arguments(sys.argv[1:])
    main(args)


if __name__ == '__main__':
    cli()
