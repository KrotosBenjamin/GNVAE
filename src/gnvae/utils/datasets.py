import os
import abc
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob

import torch
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

COLOUR_BLACK = 0
COLOUR_WHITE = 1
DIR = os.path.abspath(os.path.dirname(__file__))

DATASETS_DICT = {"geneexpression": "GeneExpression"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset):
    """Return the correct dataset class."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError(f"Unkown dataset: {dataset}")


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, drop_last=False,
                    logger=logging.getLogger(__name__), **kwargs):
    """
    A generic data loader.

    Parameters
    ----------
    dataset : {"geneexpression", ...}
        Name of the dataset to load.
    root : str, optional
        Path to the dataset root. If `None` uses the default one.
    kwargs :
        Additional arguments to the Dataset constructor and `DataLoader`.
    """
    pin_memory = pin_memory and torch.cuda.is_available()  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset_instance = Dataset(root=root or "/", logger=logger, **kwargs)
    return DataLoader(dataset_instance,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      drop_last=drop_last)


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.
    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        #self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if type(self) != GeneExpression and not os.path.isdir(root):
            self.logger.info(f"Downloading {str(type(self))} ...")
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        # This will be set by the child class
        return len(self.data)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class GeneExpression(DisentangledDataset):
    """
    Gene expression data.
    Each gene is treated as a data point, and samples are the features.

    This class can operate in two modes:
    1. Splitting Mode: If `gene_expression_filename` is given, it creates a 10-fold
       cross-validation split of the data reproducibly.
    2. Loading Mode: If `gene_expression_dir` is given, it loads pre-split
       'X_train.csv' or 'X_test.csv' files from that directory.

    Parameters
    ----------
    root : str, optional
        Root directory for the dataset (largely unused here).
    gene_expression_filename : str, optional
        Path to the full gene-by-sample CSV file. Used for splitting mode.
    gene_expression_dir : str, optional
        Path to a directory containing 'X_train.csv' and 'X_test.csv'.
        Used for loading mode.
    fold_id : int, default: 0
        The fold index (0-9) to use when in splitting mode.
    train : bool, default: True
        If True, loads the training data for the fold; otherwise, loads the test data.
    random_state : int, default: 42
        The random seed for creating reproducible CV splits.
    """
    #files = {"train": "None"}
    #img_size = (1, 20, 20)
    background_color = COLOUR_BLACK

    def __init__(self, root="/", gene_expression_filename=None,
                 gene_expression_dir=None, fold_id=0, train=True,
                 random_state=13, **kwargs):
        super().__init__(root, [], **kwargs)
        # Validate that exactly one data source is provided
        if not (gene_expression_filename or gene_expression_dir) or \
           (gene_expression_filename and gene_expression_dir):
            raise ValueError("Please provide either `gene_expression_filename` or `gene_expression_dir`.")

        # Data Loading and Splitting
        if gene_expression_filename:
            self.logger.info(f"Loading and splitting data from: {gene_expression_filename}")
            full_df = pd.read_csv(gene_expression_filename, index_col=0, sep=None,
                                  engine='python')
            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
            # Get the train/test indices for the specified fold
            all_splits = list(kf.split(full_df))
            if not (0 <= fold_id < 10):
                raise ValueError(f"fold_id must be between 0 and 9, but got {fold_id}")
            train_idx, test_idx = all_splits[fold_id]
            if train:
                self.logger.info(f"Using training data from fold {fold_id}/{kf.n_splits-1}")
                self.dfx = full_df.iloc[train_idx]
            else:
                self.logger.info(f"Using test data from fold {fold_id}/{kf.n_splits-1}")
                self.dfx = full_df.iloc[test_idx]

        else: # gene_expression_dir is provided
            self.logger.info(f"Loading pre-split data from: {gene_expression_dir}")
            file_to_load = 'X_train.csv' if train else 'X_test.csv'
            data_path = os.path.join(gene_expression_dir, file_to_load)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Could not find required file: {data_path}")
            self.dfx = pd.read_csv(data_path, index_col=0, sep=None,
                                   engine='python')

        # Data Conversion to Tensor
        self.img_size = (1, self.dfx.shape[1]) # Features = number of samples
        self.data = torch.from_numpy(self.dfx.values.astype(np.float32))

    def __getitem__(self, idx):
        """Return a single gene's expression profile as a tensor."""
        # The data is already a tensor
        item = self.data[idx]
        # No labels, so return 0 as a placeholder required by DataLoader
        return item, 0

    def download(self):
        """Not used for this dataset."""
        pass


# HELPERS
def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.
    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.
    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.
    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)
