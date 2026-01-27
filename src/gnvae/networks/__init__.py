"""Network extraction utilities for GNVAE.

This module provides functionality to derive gene-gene networks from trained
GNVAE models and cluster them into gene modules.

Network Extraction Methods
--------------------------
- **w_similarity**: Cosine similarity between decoder weight rows
- **latent_cov**: Propagate latent variance through decoder
- **graphical_lasso**: Conditional independence via Graphical Lasso

Module Clustering
-----------------
- **leiden_modules**: Leiden community detection (requires optional dependencies)
- **spectral_modules**: Spectral clustering (no extra dependencies)

Example
-------
>>> from gnvae.networks import load_model, load_decoder_weights, compute_W_similarity
>>> model = load_model("path/to/model")
>>> W = load_decoder_weights(model)
>>> adjacency = compute_W_similarity(W)

CLI Usage
---------
Extract networks::

    gnvae-networks extract-networks \\
        --model-path <dir> \\
        --dataset <csv> \\
        --output-dir <dir> \\
        --methods w_similarity latent_cov

Extract modules::

    gnvae-networks extract-modules \\
        --adjacency <path> \\
        --expr <csv> \\
        --output-dir <dir> \\
        --cluster-method leiden
"""
from gnvae.networks import extract_networks, module_extraction, utils
from gnvae.networks.extract_networks import *
from gnvae.networks.module_extraction import *
from gnvae.networks.utils import *
from gnvae.networks.cli import cli

__all__ = [  # type: ignore[var-annotated]
    *extract_networks.__all__,
    *module_extraction.__all__,
    *utils.__all__,
    "cli",
]
