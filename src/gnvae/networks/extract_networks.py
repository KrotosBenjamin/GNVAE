"""Network extraction utilities for GNVAE.

This module provides reusable helpers to derive gene-gene networks from
trained GNVAE models using several complementary strategies:

1. **Decoder-loading similarity (Method A)** - cosine similarity between rows
   of the decoder weight matrix ``W``.
2. **Latent-space covariance propagation (Method B)** - propagate posterior
   uncertainty ``diag(exp(logvar_mean))`` through the decoder.
3. **Conditional independence graph (Method C)** - fit a Graphical Lasso on
   reconstructed expression ``X_hat = Z @ W.T``.

Note: Unlike BSVAE, GNVAE does not use a Laplacian prior, so the laplacian
method is not supported.

The functions here are device-agnostic and written for integration in both CLI
workflows and unit tests.
"""
from __future__ import annotations

import gzip
import heapq
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.covariance import GraphicalLasso
from torch.utils.data import DataLoader, Dataset, TensorDataset

from gnvae.disvae.utils.modelIO import load_model as gnvae_load_model, load_metadata

logger = logging.getLogger(__name__)


@dataclass
class NetworkResults:
    """Container for multiple adjacency matrices.

    Attributes
    ----------
    method : str
        Name of the method that produced the adjacency.
    adjacency : np.ndarray
        Symmetric matrix (G, G) encoding gene-gene connectivity.
    aux : dict
        Optional auxiliary outputs such as covariance or precision matrices.
    """

    method: str
    adjacency: np.ndarray
    aux: Optional[dict] = None


def load_model(model_path: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Load a trained GNVAE model from a directory or checkpoint path.

    Parameters
    ----------
    model_path
        Path to the directory containing ``specs.json`` and ``model.pt`` or a
        direct path to the checkpoint file.
    device
        Torch device to place the model on. Defaults to CUDA when available.

    Returns
    -------
    torch.nn.Module
        Loaded model in evaluation mode.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isdir(model_path):
        directory, filename = model_path, "model.pt"
    else:
        directory, filename = os.path.dirname(model_path), os.path.basename(model_path)

    model = gnvae_load_model(directory, is_gpu=(device.type == "cuda"), filename=filename)
    model = model.to(device)
    model.eval()
    return model


def load_decoder_weights(model: torch.nn.Module) -> torch.Tensor:
    """Return the decoder weights from a GNVAE model.

    Supports both DecoderFullyconnected (output_layer.weight) and
    DecoderFullyconnected5 (lin1.weight) architectures.

    Parameters
    ----------
    model
        GNVAE model instance.

    Returns
    -------
    torch.Tensor
        Decoder weights of shape ``(G, K)`` where G is number of genes
        and K is the hidden dimension.
    """
    decoder = model.decoder

    # DecoderFullyconnected5: lin1.weight (n_genes, 128)
    # This decoder has lin0 and lin1, where lin1 maps to output
    if hasattr(decoder, 'lin1') and not hasattr(decoder, 'output_layer'):
        return decoder.lin1.weight.detach()

    # DecoderFullyconnected: output_layer.weight (n_genes, hidden_dims[-1])
    if hasattr(decoder, 'output_layer'):
        return decoder.output_layer.weight.detach()

    raise ValueError(
        f"Cannot extract weights from decoder type: {type(decoder).__name__}. "
        "Expected DecoderFullyconnected or DecoderFullyconnected5."
    )


def compute_W_similarity(W: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """Compute cosine similarity between gene loading vectors (Method A).

    Parameters
    ----------
    W
        Decoder weight matrix ``(G, K)``.
    eps
        Numerical stability constant added to norms.

    Returns
    -------
    np.ndarray
        Symmetric adjacency matrix ``(G, G)`` with cosine similarities.
    """
    W = W.float()
    W_norm = F.normalize(W, dim=1, eps=eps)
    adjacency = torch.matmul(W_norm, W_norm.T)
    return adjacency.cpu().numpy()


def compute_latent_covariance(
    W: torch.Tensor,
    logvar_mean: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate latent posterior variance through the decoder (Method B).

    Covariance is approximated as ``W @ diag(exp(logvar_mean)) @ W.T`` where
    ``logvar_mean`` is the dataset-average latent log-variance.

    Parameters
    ----------
    W
        Decoder weight matrix ``(G, K)``.
    logvar_mean
        Mean log-variance across samples ``(K,)``.
    eps
        Numerical jitter applied to the diagonal when computing correlation.

    Returns
    -------
    cov : np.ndarray
        Gene-gene covariance matrix ``(G, G)``.
    corr : np.ndarray
        Gene-gene Pearson correlation matrix ``(G, G)``.
    """
    if logvar_mean.dim() != 1:
        raise ValueError("logvar_mean must be a 1D tensor of length K")

    latent_var = torch.exp(logvar_mean)
    cov = torch.matmul(W, torch.diag(latent_var))
    cov = torch.matmul(cov, W.T)

    diag = torch.diag(cov).clamp(min=eps)
    std = torch.sqrt(diag)
    corr = cov / torch.outer(std, std)
    return cov.cpu().numpy(), corr.cpu().numpy()


def compute_graphical_lasso(
    latent_samples: np.ndarray,
    W: torch.Tensor,
    alpha: float = 0.01,
    max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Graphical Lasso on reconstructed expression (Method C).

    Parameters
    ----------
    latent_samples
        Array of latent representations ``(n_samples, K)``; typically the
        posterior means ``mu``.
    W
        Decoder weights ``(G, K)``.
    alpha
        Regularization strength for :class:`sklearn.covariance.GraphicalLasso`.
    max_iter
        Maximum number of iterations for the solver.

    Returns
    -------
    precision : np.ndarray
        Estimated precision matrix ``(G, G)``.
    covariance : np.ndarray
        Model-implied covariance matrix from the Graphical Lasso.
    adjacency : np.ndarray
        Binary adjacency where non-zero precision entries indicate edges.
    """
    Xhat = np.matmul(latent_samples, W.detach().cpu().numpy().T)
    gl = GraphicalLasso(alpha=alpha, max_iter=max_iter)
    gl.fit(Xhat)
    precision = gl.precision_
    covariance = gl.covariance_
    adjacency = (np.abs(precision) > 0).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    return precision, covariance, adjacency


def extract_latents(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract latent representations (mu, logvar) for all samples.

    Parameters
    ----------
    model
        Trained GNVAE model.
    dataloader
        DataLoader yielding ``(expression, sample_id)`` pairs.
    device
        Device for computation. If None, uses model's device.

    Returns
    -------
    mu : np.ndarray
        Array of latent means ``(n_samples, K)``.
    logvar : np.ndarray
        Array of latent log-variances ``(n_samples, K)``.
    sample_ids : list of str
        Sample identifiers in order.
    """
    device = device or next(model.parameters()).device
    model.eval()

    mu_list = []
    logvar_list = []
    sample_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, ids = batch[0], batch[1]
            else:
                data = batch
                ids = [str(i) for i in range(len(data))]

            data = data.to(device)
            mu, logvar = model.encoder(data)
            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())

            # Handle sample IDs
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy().tolist()
            sample_ids.extend([str(s) for s in ids])

    mu = np.concatenate(mu_list, axis=0)
    logvar = np.concatenate(logvar_list, axis=0)

    return mu, logvar, sample_ids


def _infer_separator(path: str) -> str:
    suffixes = [s.lower() for s in Path(path).suffixes]
    return "\t" if ".tsv" in suffixes else ","


def load_expression(path: str) -> pd.DataFrame:
    """Load a gene expression matrix (genes x samples)."""
    sep = _infer_separator(path)
    return pd.read_csv(path, index_col=0, sep=sep)


def create_dataloader_from_expression(
    path: str,
    batch_size: int = 128
) -> Tuple[DataLoader, List[str], List[str]]:
    """Create a DataLoader from a genes x samples matrix.

    Parameters
    ----------
    path
        CSV/TSV file containing the expression matrix.
    batch_size
        Batch size for the returned DataLoader.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        Yields ``(expression, sample_id)`` pairs with shape ``(batch, G)``.
    genes : list of str
        Gene identifiers from the index.
    samples : list of str
        Sample identifiers from the columns.
    """
    df = load_expression(path)
    tensor = torch.from_numpy(df.T.values.astype(np.float32))
    dataset = TensorDataset(tensor, torch.arange(tensor.shape[0]))

    class SampleIdWrapper(Dataset):
        def __init__(self, base: Dataset, sample_ids: Sequence[str]):
            self.base = base
            self.sample_ids = list(sample_ids)

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, _ = self.base[idx]
            return x, self.sample_ids[idx]

    wrapped = SampleIdWrapper(dataset, df.columns)
    dataloader = DataLoader(wrapped, batch_size=batch_size, shuffle=False)
    return dataloader, list(df.index), list(df.columns)


def save_adjacency_matrix(
    adjacency: np.ndarray,
    output_path: str,
    genes: Optional[Sequence[str]] = None
) -> None:
    """Persist an adjacency matrix to disk.

    Parameters
    ----------
    adjacency
        Square matrix to save.
    output_path
        Destination path. ``.csv``/``.tsv`` are written via pandas, otherwise
        ``.npy`` is used.
    genes
        Optional gene identifiers to use as row/column labels.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        n_genes = adjacency.shape[0]
        if genes is None:
            genes = [str(i) for i in range(n_genes)]
        with open(path, "w") as f:
            f.write(f"gene{sep}{sep.join(str(g) for g in genes)}\n")
            for i in range(n_genes):
                row_values = sep.join(str(v) for v in adjacency[i])
                f.write(f"{genes[i]}{sep}{row_values}\n")
    else:
        np.save(path, adjacency)


def save_edge_list(
    adjacency: np.ndarray,
    output_path: str,
    genes: Optional[Sequence[str]] = None,
    threshold: float = 0.0,
    include_self: bool = False
) -> None:
    """Save an adjacency matrix as an edge list.

    Parameters
    ----------
    adjacency
        Square matrix encoding weights.
    output_path
        CSV/TSV path for the edge list.
    genes
        Optional list of gene names; defaults to integer indices.
    threshold
        Minimum absolute weight to keep an edge.
    include_self
        Whether to keep self-loops.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if genes is None:
        genes = list(range(adjacency.shape[0]))
    genes = list(genes)

    sep = "," if path.suffix.lower() == ".csv" else "\t"
    with open(path, "w") as f:
        f.write(f"source{sep}target{sep}weight\n")
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[1]):
                if not include_self and i == j:
                    continue
                weight = adjacency[i, j]
                if abs(weight) >= threshold:
                    f.write(f"{genes[i]}{sep}{genes[j]}{sep}{weight}\n")


def _iter_upper_triangle(
    adjacency: np.ndarray,
    threshold: float = 0.0,
    include_diagonal: bool = False,
) -> Iterator[Tuple[int, int, float]]:
    """Stream upper-triangle edges without materializing the full triangle.

    Yields (i, j, weight) tuples for edges with |weight| >= threshold.
    Iterates row-by-row to avoid O(n^2) memory allocation.
    """
    n = adjacency.shape[0]
    for i in range(n):
        start_j = i if include_diagonal else i + 1
        row = adjacency[i, start_j:]
        if threshold > 0:
            mask = np.abs(row) >= threshold
            cols = np.where(mask)[0] + start_j
            for j in cols:
                yield i, j, float(adjacency[i, j])
        else:
            for offset, w in enumerate(row):
                yield i, start_j + offset, float(w)


def compute_adaptive_threshold(adjacency: np.ndarray, target_sparsity: float = 0.01) -> float:
    """Compute threshold to achieve target sparsity in the adjacency matrix.

    Uses a streaming heap-based algorithm to avoid materializing the full
    upper triangle, which is critical for large networks.

    Parameters
    ----------
    adjacency
        Square adjacency matrix.
    target_sparsity
        Target fraction of edges to keep (e.g., 0.01 = top 1%).

    Returns
    -------
    float
        Threshold value such that keeping edges >= threshold yields target sparsity.
    """
    n = adjacency.shape[0]
    n_total_edges = n * (n - 1) // 2
    n_edges_target = max(1, int(n_total_edges * target_sparsity))

    if n_edges_target >= n_total_edges:
        return 0.0

    top_k: List[float] = []

    for i in range(n):
        row = np.abs(adjacency[i, i + 1:])
        for w in row:
            if len(top_k) < n_edges_target:
                heapq.heappush(top_k, w)
            elif w > top_k[0]:
                heapq.heapreplace(top_k, w)

    return float(top_k[0]) if top_k else 0.0


def save_adjacency_sparse(
    adjacency: np.ndarray,
    output_path: str,
    genes: Optional[Sequence[str]] = None,
    threshold: float = 0.0,
    compress: bool = True,
    quantize: Union[bool, str] = True,
) -> None:
    """Save an adjacency matrix in compressed format.

    For symmetric matrices, only the upper triangle values are stored (no indices
    needed for dense matrices), reducing size by ~50%. Combined with quantization
    and gzip compression, a 20k gene network can be stored in ~50-80 MB.

    Parameters
    ----------
    adjacency
        Square matrix to save.
    output_path
        Destination path. Should end with ``.npz``.
    genes
        Optional gene identifiers to save alongside the matrix.
    threshold
        Minimum absolute weight to keep an edge. Edges below threshold are zeroed.
    compress
        Whether to use compression (default True).
    quantize
        Quantization level: True/"float16", "int8", or False/"float32".
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = adjacency.shape[0]

    if genes is None:
        genes = [str(i) for i in range(n)]
    genes_arr = np.array(genes, dtype=object)

    save_func = np.savez_compressed if compress else np.savez

    # Determine dtype name
    if quantize == "int8":
        dtype_name = "int8"
    elif quantize is False or quantize == "float32":
        dtype_name = "float32"
    else:
        dtype_name = "float16"

    # When threshold > 0, use streaming to avoid materializing full upper triangle
    if threshold > 0:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for i, j, w in _iter_upper_triangle(adjacency, threshold=threshold, include_diagonal=True):
            rows.append(i)
            cols.append(j)
            data.append(w)

        n_nonzero = len(data)
        n_total = n * (n + 1) // 2
        density = n_nonzero / n_total

        data_arr = np.array(data, dtype=np.float32)
        scale_factor = None
        if quantize == "int8":
            scale_factor = np.abs(data_arr).max() if len(data_arr) > 0 else 0.0
            if scale_factor > 0:
                data_q = (data_arr / scale_factor * 127).astype(np.int8)
            else:
                data_q = np.zeros(len(data_arr), dtype=np.int8)
        elif quantize is False or quantize == "float32":
            data_q = data_arr
        else:
            data_q = data_arr.astype(np.float16)

        idx_dtype = np.int16 if n < 32768 else np.int32

        save_dict = {
            "data": data_q,
            "row": np.array(rows, dtype=idx_dtype),
            "col": np.array(cols, dtype=idx_dtype),
            "n": np.array([n], dtype=np.int32),
            "genes": genes_arr,
            "storage_format": np.array(["sparse_triu"], dtype=object),
            "dtype": np.array([dtype_name], dtype=object),
        }
        if scale_factor is not None:
            save_dict["scale_factor"] = np.array([scale_factor], dtype=np.float32)

        save_func(path, **save_dict)
        logger.info(
            "Saved sparse adjacency to %s: %d genes, %d edges (%.1f%% density), dtype=%s",
            path, n, n_nonzero, density * 100, dtype_name,
        )
        return

    # threshold = 0: need full upper triangle for dense storage
    triu_idx = np.triu_indices(n)
    triu_data = adjacency[triu_idx]

    n_nonzero = np.sum(triu_data != 0)
    n_total = len(triu_data)
    density = n_nonzero / n_total

    scale_factor = None
    if quantize == "int8":
        vmin, vmax = triu_data.min(), triu_data.max()
        scale_factor = max(abs(vmin), abs(vmax))
        if scale_factor > 0:
            triu_data_q = (triu_data / scale_factor * 127).astype(np.int8)
        else:
            triu_data_q = np.zeros_like(triu_data, dtype=np.int8)
    elif quantize is False or quantize == "float32":
        triu_data_q = triu_data.astype(np.float32)
    else:
        triu_data_q = triu_data.astype(np.float16)

    if density > 0.5:
        save_dict = {
            "triu_values": triu_data_q,
            "n": np.array([n], dtype=np.int32),
            "genes": genes_arr,
            "storage_format": np.array(["dense_triu"], dtype=object),
            "dtype": np.array([dtype_name], dtype=object),
        }
        if scale_factor is not None:
            save_dict["scale_factor"] = np.array([scale_factor], dtype=np.float32)

        save_func(path, **save_dict)
        logger.info(
            "Saved dense adjacency to %s: %d genes, %.1f%% density, dtype=%s",
            path, n, density * 100, dtype_name,
        )
    else:
        nonzero_mask = triu_data != 0
        row = triu_idx[0][nonzero_mask]
        col = triu_idx[1][nonzero_mask]

        if quantize == "int8" and scale_factor is not None and scale_factor > 0:
            data_q = (triu_data[nonzero_mask] / scale_factor * 127).astype(np.int8)
        elif quantize is False or quantize == "float32":
            data_q = triu_data[nonzero_mask].astype(np.float32)
        else:
            data_q = triu_data[nonzero_mask].astype(np.float16)

        idx_dtype = np.int16 if n < 32768 else np.int32

        save_dict = {
            "data": data_q,
            "row": row.astype(idx_dtype),
            "col": col.astype(idx_dtype),
            "n": np.array([n], dtype=np.int32),
            "genes": genes_arr,
            "storage_format": np.array(["sparse_triu"], dtype=object),
            "dtype": np.array([dtype_name], dtype=object),
        }
        if scale_factor is not None:
            save_dict["scale_factor"] = np.array([scale_factor], dtype=np.float32)

        save_func(path, **save_dict)
        logger.info(
            "Saved sparse adjacency to %s: %d genes, %d edges (%.1f%% density), dtype=%s",
            path, n, n_nonzero, density * 100, dtype_name,
        )


def load_adjacency_sparse(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load an adjacency matrix from NPZ format.

    Handles multiple storage formats:
    - dense_triu: Upper triangle values stored as flat array
    - sparse_triu: Upper triangle with indices

    Returns
    -------
    adjacency : np.ndarray
        Dense adjacency matrix reconstructed from storage.
    genes : list[str]
        Gene identifiers.
    """
    data = np.load(path, allow_pickle=True)
    genes = list(data["genes"])

    storage_format = None
    if "storage_format" in data:
        storage_format = str(data["storage_format"][0])

    scale_factor = None
    if "scale_factor" in data:
        scale_factor = float(data["scale_factor"][0])

    def dequantize(values):
        if scale_factor is not None and values.dtype == np.int8:
            return (values.astype(np.float32) / 127.0) * scale_factor
        return values.astype(np.float32)

    if storage_format == "dense_triu":
        n = int(data["n"][0])
        triu_values = dequantize(data["triu_values"])

        adjacency = np.zeros((n, n), dtype=np.float32)
        triu_idx = np.triu_indices(n)
        adjacency[triu_idx] = triu_values
        adjacency = adjacency + adjacency.T - np.diag(np.diag(adjacency))

    elif storage_format == "sparse_triu":
        n = int(data["n"][0])
        row = data["row"]
        col = data["col"]
        values = dequantize(data["data"])

        adjacency = np.zeros((n, n), dtype=np.float32)
        adjacency[row, col] = values
        off_diag_mask = row != col
        adjacency[col[off_diag_mask], row[off_diag_mask]] = values[off_diag_mask]

    elif "shape" in data:
        shape = tuple(data["shape"])
        row = data["row"]
        col = data["col"]
        values = dequantize(data["data"])

        if "upper_triangle" in data and data["upper_triangle"][0]:
            n = shape[0]
            adjacency = np.zeros(shape, dtype=np.float32)
            adjacency[row, col] = values
            off_diag_mask = row != col
            adjacency[col[off_diag_mask], row[off_diag_mask]] = values[off_diag_mask]
        else:
            sparse_adj = sp.coo_matrix((values, (row, col)), shape=shape)
            adjacency = sparse_adj.toarray().astype(np.float32)
    else:
        raise ValueError(f"Unknown adjacency storage format in {path}")

    logger.info("Loaded adjacency from %s: %d genes", path, len(genes))
    return adjacency, genes


def save_edge_list_parquet(
    adjacency: np.ndarray,
    output_path: str,
    genes: Optional[Sequence[str]] = None,
    threshold: float = 0.0,
    include_self: bool = False,
    compression: str = "zstd",
) -> None:
    """Save an adjacency matrix as a Parquet edge list.

    Parquet format provides better compression and faster read/write performance
    compared to gzipped CSV. Gene names are stored as metadata within the file.

    Parameters
    ----------
    adjacency
        Square matrix encoding weights.
    output_path
        Output path. Should end with ``.parquet``.
    genes
        Optional list of gene names.
    threshold
        Minimum absolute weight to keep an edge.
    include_self
        Whether to keep self-loops.
    compression
        Parquet compression codec: "zstd", "snappy", "gzip", or None.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not installed, falling back to CSV edge list")
        csv_path = str(output_path).replace(".parquet", ".csv")
        save_edge_list(adjacency, csv_path, genes, threshold, include_self)
        return

    path = Path(output_path)
    if not str(path).endswith(".parquet"):
        path = Path(str(path).replace(".csv", "").replace(".tsv", "") + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)

    if genes is None:
        genes = [str(i) for i in range(adjacency.shape[0])]
    genes = list(genes)

    src_list: List[int] = []
    tgt_list: List[int] = []
    weight_list: List[float] = []

    for i, j, w in _iter_upper_triangle(adjacency, threshold=threshold, include_diagonal=include_self):
        src_list.append(i)
        tgt_list.append(j)
        weight_list.append(w)

    df = pd.DataFrame({
        "source_idx": np.array(src_list, dtype=np.int32),
        "target_idx": np.array(tgt_list, dtype=np.int32),
        "weight": np.array(weight_list, dtype=np.float32),
    })

    table = pa.Table.from_pandas(df, preserve_index=False)
    genes_json = "\n".join(genes)
    metadata = {
        b"genes": genes_json.encode("utf-8"),
        b"n_genes": str(len(genes)).encode("utf-8"),
    }
    existing_metadata = table.schema.metadata or {}
    merged_metadata = {**existing_metadata, **metadata}
    table = table.replace_schema_metadata(merged_metadata)

    pq.write_table(table, path, compression=compression)

    logger.info(
        "Saved edge list to %s: %d edges (threshold=%.4f, compression=%s)",
        path, len(df), threshold, compression,
    )


def load_edge_list_parquet(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load an adjacency matrix from a Parquet edge list.

    Returns
    -------
    adjacency : np.ndarray
        Dense adjacency matrix reconstructed from edge list.
    genes : list[str]
        Gene identifiers extracted from file metadata.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required to load Parquet edge lists")

    path = Path(path)
    table = pq.read_table(path)
    df = table.to_pandas()

    metadata = table.schema.metadata or {}
    if b"genes" in metadata:
        genes_str = metadata[b"genes"].decode("utf-8")
        genes = genes_str.split("\n")
    else:
        max_idx = max(df["source_idx"].max(), df["target_idx"].max())
        genes = [str(i) for i in range(max_idx + 1)]
        logger.warning("No genes metadata found in %s, using numeric indices", path)

    n = len(genes)
    adjacency = np.zeros((n, n), dtype=np.float32)

    src_idx = df["source_idx"].values
    tgt_idx = df["target_idx"].values
    weights = df["weight"].values

    adjacency[src_idx, tgt_idx] = weights
    adjacency[tgt_idx, src_idx] = weights

    logger.info("Loaded edge list from %s: %d genes, %d edges", path, n, len(df))
    return adjacency, genes


def run_extraction(
    model: torch.nn.Module,
    dataloader: DataLoader,
    genes: Sequence[str],
    methods: Iterable[str],
    threshold: float = 0.0,
    alpha: float = 0.01,
    output_dir: Optional[str] = None,
    create_heatmaps: bool = False,
    sparse: bool = True,
    compress: bool = True,
    target_sparsity: float = 0.01,
    quantize: Union[bool, str] = "int8",
) -> List[NetworkResults]:
    """Run requested network extraction methods.

    Parameters
    ----------
    model
        Loaded GNVAE model.
    dataloader
        Iterator over expression data.
    genes
        Gene identifiers corresponding to decoder rows.
    methods
        Iterable of methods to compute: "w_similarity", "latent_cov", "graphical_lasso".
    threshold
        Threshold applied when writing edge lists. If 0 and sparse=True,
        an adaptive threshold is computed based on target_sparsity.
    alpha
        Graphical Lasso regularization strength.
    output_dir
        Optional directory to persist results.
    create_heatmaps
        When ``True`` generate matplotlib heatmaps for adjacencies.
    sparse
        When ``True`` (default), save adjacency in sparse NPZ format.
    compress
        When ``True`` (default), use compression.
    target_sparsity
        Target fraction of edges to keep when using adaptive thresholding.
    quantize
        Quantization level: "int8" (smallest), "float16", or "float32".

    Returns
    -------
    list of NetworkResults
        One entry per computed method.
    """
    device = next(model.parameters()).device
    W = load_decoder_weights(model).to(device)
    methods = [m.lower() for m in methods]

    if W.shape[0] != len(genes):
        raise ValueError(
            f"Gene dimension mismatch: decoder has {W.shape[0]} rows but {len(genes)} genes were provided."
        )

    mu, logvar, sample_ids = extract_latents(model, dataloader, device=device)
    results: List[NetworkResults] = []

    if "w_similarity" in methods:
        adjacency = compute_W_similarity(W)
        results.append(NetworkResults("w_similarity", adjacency))
        _persist(adjacency, genes, output_dir, "w_similarity", threshold, create_heatmaps, sparse, compress, target_sparsity, quantize)

    if "latent_cov" in methods:
        logvar_mean = torch.from_numpy(logvar).to(device).mean(dim=0)
        cov, corr = compute_latent_covariance(W, logvar_mean)
        results.append(NetworkResults("latent_cov", cov, {"correlation": corr}))
        _persist(cov, genes, output_dir, "latent_cov", threshold, create_heatmaps, sparse, compress, target_sparsity, quantize)
        if output_dir:
            _persist(corr, genes, output_dir, "latent_cov_correlation", threshold, False, sparse, compress, target_sparsity, quantize)

    if "graphical_lasso" in methods:
        precision, covariance, adjacency = compute_graphical_lasso(mu, W, alpha=alpha)
        results.append(NetworkResults("graphical_lasso", adjacency, {"precision": precision, "covariance": covariance}))
        _persist(adjacency, genes, output_dir, "graphical_lasso", threshold, create_heatmaps, sparse, compress, target_sparsity, quantize)
        if output_dir:
            _persist(precision, genes, output_dir, "graphical_lasso_precision", threshold, False, sparse, compress, target_sparsity, quantize)

    return results


def _persist(
    adjacency: np.ndarray,
    genes: Sequence[str],
    output_dir: Optional[str],
    prefix: str,
    threshold: float,
    create_heatmaps: bool,
    sparse: bool = True,
    compress: bool = True,
    target_sparsity: float = 0.01,
    quantize: Union[bool, str] = "int8",
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)

    edge_threshold = threshold
    if sparse and threshold == 0.0:
        edge_threshold = compute_adaptive_threshold(adjacency, target_sparsity)
        logger.info(
            "Using adaptive threshold %.4f for %s edge list (target sparsity=%.1f%%)",
            edge_threshold, prefix, target_sparsity * 100,
        )

    if sparse:
        save_adjacency_sparse(
            adjacency,
            os.path.join(output_dir, f"{prefix}_adjacency.npz"),
            genes,
            threshold=0.0,
            compress=compress,
            quantize=quantize,
        )
        compression = "zstd" if compress else None
        save_edge_list_parquet(
            adjacency,
            os.path.join(output_dir, f"{prefix}_edges.parquet"),
            genes,
            threshold=edge_threshold,
            compression=compression,
        )
    else:
        save_adjacency_matrix(adjacency, os.path.join(output_dir, f"{prefix}_adjacency.csv"), genes)
        save_edge_list(adjacency, os.path.join(output_dir, f"{prefix}_edges.csv"), genes, threshold=threshold)

    if create_heatmaps:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(adjacency, ax=ax, xticklabels=False, yticklabels=False, cmap="viridis")
            ax.set_title(prefix)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{prefix}_heatmap.png"), dpi=200)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Could not create heatmap for %s: %s", prefix, exc)


__all__ = [
    "NetworkResults",
    "load_model",
    "load_decoder_weights",
    "compute_W_similarity",
    "compute_latent_covariance",
    "compute_graphical_lasso",
    "extract_latents",
    "save_adjacency_matrix",
    "save_edge_list",
    "compute_adaptive_threshold",
    "save_adjacency_sparse",
    "load_adjacency_sparse",
    "save_edge_list_parquet",
    "load_edge_list_parquet",
    "load_expression",
    "create_dataloader_from_expression",
    "run_extraction",
]
