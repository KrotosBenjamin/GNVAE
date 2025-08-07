#!/usr/bin/env python

## Original author: Apuã Paquola
## Modified by: Javier Hernandez and Kynon J.M. Benjamin

import os
import sys
import argparse
import logging

import torch
import numpy as np
import pandas as pd
from gnvae.utils.datasets import get_dataloaders
from gnvae.disvae.utils.modelIO import load_model, load_metadata

def extract_latent_variables(model, dataloader, device):
    """Yields: np.array with [mu_1 ... mu_n, logvar_1 ... logvar_n] for each batch"""
    model.eval()
    for data, _ in dataloader:
        data = data.to(device)
        with torch.no_grad():
            mu, logvar = model.encoder(data)
            mu = mu.cpu().numpy().squeeze()
            logvar = logvar.cpu().numpy().squeeze()
            result = np.concatenate([mu, logvar])
            yield result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-filename', required=True,
                        help="Path to model .pt file.")
    parser.add_argument('--gene-expression-filename',
                        help="Path to gene expression TSV/CSV.")
    parser.add_argument('--gene-expression-dir',
                        help="Directory with 'X_train.csv' as gene expression data.")
    parser.add_argument('--output-filename', default="./output",
                        help="Output TSV path.")
    parser.add_argument('--use-cuda', action='store_true',
                        help="Force use of CUDA if available.")
    args = parser.parse_args()

    if (args.gene_expression_filename is None) == (args.gene_expression_dir is None):
        raise ValueError("You must specify exactly one of --gene-expression-filename or --gene-expression-dir")

    gene_expression_path = args.gene_expression_filename
    if args.gene_expression_dir:
        gene_expression_path = os.path.join(args.gene_expression_dir, "X_train.csv")

    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    logging.info("Loading model...")
    model_dir = os.path.dirname(args.model_filename)
    model = load_model(model_dir, filename=os.path.basename(args.model_filename))
    model = model.to(device)
    metadata = load_metadata(model_dir)
    logging.info("... done.")

    logging.info("Reading expression file index...")
    # read index; flexible parsing for sep
    df_index = pd.read_csv(gene_expression_path, sep=None, engine='python',
                           index_col=0, usecols=[0])
    logging.info("... done.")

    logging.info("Preparing dataloader...")
    dataset_kwargs = {
        'gene_expression_filename': args.gene_expression_filename,
        'gene_expression_dir': args.gene_expression_dir,
        'fold_id': "alldata"
    }
    dataloader = get_dataloaders(
        "geneexpression",
        shuffle=False,
        batch_size=1,
        **{k: v for k, v in dataset_kwargs.items() if v is not None}
    )

    logging.info("Calculating latent variables...")
    latent_records = list(extract_latent_variables(model, dataloader, device))
    df_latent = pd.DataFrame(latent_records, index=df_index.index)
    assert df_latent.shape[0] == df_index.shape[0], "Sample count across expression and encoding mismatch."

    n_latent = df_latent.shape[1] // 2
    df_latent.columns = (
        [f'mu{i}' for i in range(n_latent)]
        + [f'logvar{i}' for i in range(n_latent)]
    )
    df_latent.index.name = df_index.index.name

    logging.info("Saving latent variables...")
    df_latent.to_csv(args.output_filename, sep="\t")
    logging.info(f"... done. TSV saved to {args.output_filename}")


if __name__ == '__main__':
    main()
