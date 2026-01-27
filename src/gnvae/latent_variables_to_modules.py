#!/usr/bin/env python
"""Simplified CLI for extracting gene modules from GNVAE networks.

This module provides a thin CLI wrapper around gnvae.networks for module extraction.
For full functionality, use the gnvae-networks CLI instead.

Example usage::

    gnvae-modules --adjacency network.npz --expr expression.csv --output-dir ./modules

For more options, use the gnvae-networks extract-modules command::

    gnvae-networks extract-modules --help
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gnvae.networks.module_extraction import (
    leiden_modules,
    spectral_modules,
    compute_module_eigengenes,
    load_adjacency,
    save_modules,
    save_eigengenes,
)
from gnvae.networks.extract_networks import (
    load_model,
    load_expression,
    run_extraction,
    create_dataloader_from_expression,
    save_adjacency_sparse,
)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        level=getattr(logging, level.upper(), logging.INFO)
    )
    return logging.getLogger(__name__)


def main():
    """Main entry point for gnvae-modules CLI."""
    parser = argparse.ArgumentParser(
        prog="gnvae-modules",
        description="Extract gene modules from GNVAE adjacency matrices."
    )

    # Input options (at least one required)
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument(
        "--adjacency",
        help="Path to adjacency matrix (.npz, .parquet, or .csv/.tsv)"
    )
    input_group.add_argument(
        "--model-path",
        help="Path to trained GNVAE model directory (alternative to --adjacency)"
    )
    input_group.add_argument(
        "--dataset",
        help="Gene expression matrix for network extraction (required if using --model-path)"
    )

    # Required
    parser.add_argument(
        "--expr",
        required=True,
        help="Expression matrix (genes x samples) for eigengene computation"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for modules and eigengenes"
    )

    # Clustering options
    cluster_group = parser.add_argument_group("Clustering options")
    cluster_group.add_argument(
        "--cluster-method",
        choices=["leiden", "spectral"],
        default="leiden",
        help="Clustering algorithm (default: leiden)"
    )
    cluster_group.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution parameter (default: 1.0)"
    )
    cluster_group.add_argument(
        "--n-clusters",
        type=int,
        help="Number of clusters for spectral clustering (default: sqrt(n_genes))"
    )
    cluster_group.add_argument(
        "--adjacency-mode",
        choices=["wgcna-signed", "signed"],
        default="wgcna-signed",
        help="How to handle negative edges (default: wgcna-signed)"
    )

    # Network extraction options (when using --model-path)
    network_group = parser.add_argument_group("Network extraction options")
    network_group.add_argument(
        "--network-method",
        default="w_similarity",
        help="Method for network extraction (default: w_similarity)"
    )
    network_group.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128)"
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Validate inputs
    if not args.adjacency and not args.model_path:
        parser.error("Either --adjacency or --model-path must be provided")
    if args.model_path and not args.dataset:
        parser.error("--dataset is required when using --model-path")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or compute adjacency
    if args.adjacency:
        logger.info("Loading adjacency from %s", args.adjacency)
        adjacency, genes = load_adjacency(args.adjacency)
    else:
        logger.info("Computing adjacency from model at %s", args.model_path)
        model = load_model(args.model_path)
        dataloader, genes, _ = create_dataloader_from_expression(
            args.dataset, batch_size=args.batch_size
        )
        results = run_extraction(
            model=model,
            dataloader=dataloader,
            genes=genes,
            methods=[args.network_method],
            output_dir=None,
        )
        if not results:
            logger.error("No adjacency matrices were produced")
            sys.exit(1)
        adjacency = results[0].adjacency

        # Save the computed adjacency
        adj_path = output_dir / f"{args.network_method}_adjacency.npz"
        save_adjacency_sparse(adjacency, str(adj_path), genes, threshold=0.0, compress=True)
        logger.info("Saved computed adjacency to %s", adj_path)

    # Run clustering
    import pandas as pd
    adjacency_df = pd.DataFrame(adjacency, index=genes, columns=genes)

    if args.cluster_method == "leiden":
        logger.info(
            "Running Leiden clustering (resolution=%.2f, adjacency_mode=%s)",
            args.resolution, args.adjacency_mode
        )
        try:
            modules = leiden_modules(
                adjacency_df,
                resolution=args.resolution,
                adjacency_mode=args.adjacency_mode,
            )
        except ImportError as e:
            logger.error(
                "Leiden clustering requires optional dependencies. "
                "Install with: pip install gnvae[clustering]"
            )
            logger.error("Error: %s", e)
            sys.exit(1)
    else:
        logger.info("Running spectral clustering (n_clusters=%s)", args.n_clusters)
        modules = spectral_modules(
            adjacency_df,
            n_clusters=args.n_clusters,
        )

    # Compute eigengenes
    logger.info("Loading expression data from %s", args.expr)
    expr_df = load_expression(args.expr)
    logger.info("Computing module eigengenes...")
    eigengenes = compute_module_eigengenes(expr_df, modules)

    # Save results
    modules_path = output_dir / "modules.csv"
    eigengenes_path = output_dir / "eigengenes.csv"

    save_modules(modules, str(modules_path))
    save_eigengenes(eigengenes, str(eigengenes_path))

    logger.info("Module extraction complete!")
    logger.info("  - Modules: %s", modules_path)
    logger.info("  - Eigengenes: %s", eigengenes_path)
    logger.info("  - Number of modules: %d", modules.nunique())
    logger.info("  - Number of genes: %d", len(modules))


if __name__ == "__main__":
    main()
