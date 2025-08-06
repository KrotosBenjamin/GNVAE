import os
import abc
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

COLOUR_BLACK = 0
COLOUR_WHITE = 1
DIR = os.path.abspath(os.path.dirname(__file__))

DATASETS_DICT = {"geneexpression": "GeneExpression"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset = Dataset(logger=logger, **kwargs) if root is None else Dataset(root=root, logger=logger, **kwargs)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory)


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
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

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
    """Gene expression data. Features are samples, examples are genes"""

    files = {"train": "None"}
    #img_size = (1, 20, 20)
    background_color = COLOUR_BLACK

    def __init__(self, root="/", gene_expression_filename=None, **kwargs):
        super().__init__(root, [], **kwargs)
        self.gene_expression_filename = gene_expression_filename
        dfx = pd.read_csv(self.gene_expression_filename, index_col=0,
                          sep=None, engine='python')
        self.dfx = dfx
        self.img_size = (1, 1, dfx.shape[1])
        padding = np.product(self.img_size) - dfx.shape[1]

## PAREI AQUI
##apua@qvm8:/ceph/users/apua/projects/gnvae/test/_o_m$ python ../../code/_h/disentangling-vae/main.py myname -x factor_geneexpression -m Fullyconnected5 --dataset geneexpression --gene-expression-filename /ceph/projects/v4_phase3_paper/analysis/gnvae/input/select_samples/prep_data_cv/_m/Fold-alldata/X_train.csv
##
##
        self.imgs = np.concatenate(
            [dfx.values.astype(np.float32),
             np.zeros((dfx.shape[0], padding), dtype=np.float32)],
             axis =1).reshape((-1,
                               *self.img_size))

    def __getitem__(self, idx):

        img = torch.from_numpy(self.imgs[idx])
        #img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0

    def download(self):
        """Download the dataset."""
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
