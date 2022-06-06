#!/usr/bin/env python

import sys
import argparse
import os
import logging

import pandas as pd
import numpy as np
import torch
from disvae.utils.modelIO import load_model, load_metadata
from utils.datasets import get_dataloaders

__author__ = 'Apuã Paquola'


def latent_variables(model, dl):
    model.eval()
    for data, _ in dl:
        mu, logvar = model.encoder(data.to("cuda"))
        yield np.concatenate(
            (mu.cpu().detach().numpy()[0],
             logvar.cpu().detach().numpy()[0],))
             
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-filename', required=True)
    parser.add_argument('--gene-expression-filename', required=True)
    parser.add_argument('--output-filename', required=True)
    args=parser.parse_args()

    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG)
    
    logging.info("Loading model...")
    model_dir = os.path.dirname(args.model_filename)
    model_filename = os.path.basename(args.model_filename)
    model = load_model(model_dir, filename = model_filename)
    metadata = load_metadata(model_dir)
    logging.info("... done.")


    logging.info("Loading expression data index...")
    dfz = pd.read_csv(args.gene_expression_filename, index_col=0, usecols=[0])
    logging.info("... done.")

    logging.info("Calculating latent variables...")
    dl = get_dataloaders("geneexpression",
                         gene_expression_filename=args.gene_expression_filename,
                         shuffle=False,
                         batch_size=1)
    
    df = pd.DataFrame.from_records((latent_variables(model, dl)), index=dfz.index)

    assert len(df.columns) % 2 == 0
    nvar = len(df.columns) // 2

    df.columns = [f'mu{x}' for x in range(nvar)] + [f'logvar{x}' for x in range(nvar)]
    df.index.name = 'gene_id'
    
    logging.info("... done.")
    
    logging.info("Saving latent variables...")
    df.to_csv(args.output_filename, sep="\t")
    logging.info("... done.")
    
    
if __name__ == '__main__':
    main()

