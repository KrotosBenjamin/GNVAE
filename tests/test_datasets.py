import os
import sys
import pathlib
import tempfile
import unittest

import importlib.machinery
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import KFold
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without sklearn
    import types

    def _fallback_kfold_indices(n_splits, shuffle, random_state, n_samples):
        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indices)
        for fold_indices in np.array_split(indices, n_splits):
            train_mask = np.ones_like(indices, dtype=bool)
            train_mask[fold_indices] = False
            yield indices[train_mask], fold_indices

    class KFold:
        def __init__(self, n_splits=5, shuffle=False, random_state=None):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, X):
            n_samples = len(X)
            return _fallback_kfold_indices(
                self.n_splits, self.shuffle, self.random_state, n_samples
            )

    model_selection = types.ModuleType("sklearn.model_selection")
    model_selection.__spec__ = importlib.machinery.ModuleSpec(
        "sklearn.model_selection", loader=None
    )
    model_selection.KFold = KFold
    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    sklearn_module.model_selection = model_selection
    sys.modules.setdefault("sklearn", sklearn_module)
    sys.modules.setdefault("sklearn.model_selection", model_selection)


# Ensure the package is importable when running tests from the repository root
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gnvae.utils.datasets import GeneExpression


def _build_expression_frame(num_genes=20, num_samples=5):
    data = {
        f"sample_{col}": [float(row + col) for row in range(num_genes)]
        for col in range(num_samples)
    }
    index = [f"gene_{i}" for i in range(num_genes)]
    return pd.DataFrame(data, index=index)


class GeneExpressionTests(unittest.TestCase):
    def test_loads_compressed_tsv_file_with_expected_split(self):
        df = _build_expression_frame()

        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = os.path.join(tmpdir, "gene_expression.tsv.gz")
            df.to_csv(tsv_path, sep="\t", compression="gzip")

            gene_expression = GeneExpression(
                gene_expression_filename=tsv_path,
                fold_id=1,
                train=True,
                random_state=0,
                root="/",
            )

            kf = KFold(n_splits=10, shuffle=True, random_state=0)
            train_indices, _ = list(kf.split(df))[1]
            pd.testing.assert_frame_equal(gene_expression.dfx, df.iloc[train_indices])

    def test_finds_compressed_split_files(self):
        train_df = _build_expression_frame(num_genes=6, num_samples=2)
        test_df = _build_expression_frame(num_genes=4, num_samples=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "X_train.csv.gz")
            test_path = os.path.join(tmpdir, "X_test.csv.gz")

            train_df.to_csv(train_path, compression="gzip")
            test_df.to_csv(test_path, compression="gzip")

            train_loader = GeneExpression(gene_expression_dir=tmpdir, train=True, root="/")
            test_loader = GeneExpression(gene_expression_dir=tmpdir, train=False, root="/")

            pd.testing.assert_frame_equal(train_loader.dfx, train_df)
            pd.testing.assert_frame_equal(test_loader.dfx, test_df)
