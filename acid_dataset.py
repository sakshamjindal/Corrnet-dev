# skeleton class for dataloader
# import sys
# sys.path.append('./src')

import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, Dataset
import h5py
import tables as pytables
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import os.path as osp
import glob
import time

class AcidPairDataset(Dataset):
    def __init__(self, root_dir, mode = "val", categories = [], transform=False, pair_count = 0, object_count = 0, all_points=False):

        self.files = []
        self.all_points = all_points
        self.mode = mode

        for category in categories:
            self.files.extend(glob.glob(os.path.join(root_dir, category, "*.npy")))

        # make train and test split with seed
        np.random.seed(0)
        np.random.shuffle(self.files)
        ratio = 0.7
        self.train_files = self.files[:int(ratio * len(self.files))]
        self.test_files = self.files[int(ratio * len(self.files)):]
        
        if self.mode == "train":
            self.files = self.train_files
            self.transform = transform
            if object_count > 0:
                self.files = self.files[:object_count]
        elif self.mode == "val":
            self.files = self.test_files
            self.transform = False
            if object_count > 0:
                self.files = self.files[:5]
        else:
            raise ValueError("mode must be either train or val")
        
        self.data = []
        for i in range(len(self.files)):
            d = np.load(self.files[i], allow_pickle=True).item()
            points = d["all_points"]
            np.random.seed(0)
            idxs = np.random.choice(len(points), 1024, replace=False)
            points = points[idxs]
            self.data.append(points)

        self.pairs = [(self.data[i], self.data[j]) for i in range(len(self.data)) for j in range(len(self.data))
                       if i != j]

        np.random.seed(0)
        np.random.shuffle(self.pairs)
        if pair_count > 0:
            self.pairs = self.pairs[:pair_count]


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        pc1 = self.pairs[idx][0]
        pc2 = self.pairs[idx][1]

        # generate 1024 random indices
        # if self.mode == "val":
        #     np.random.seed(0)
        #     idxs = np.random.choice(len(pc1), 1024, replace=False)
        # else:
        #     idxs = np.random.choice(len(pc1), 1024, replace=False)
        # pc1 = pc1[idxs]
        # pc2 = pc2[idxs]

        if self.mode == "val":
            np.random.seed(0)
            idxs = np.arange(len(pc2))
            np.random.shuffle(idxs)
        else:
            idxs = np.arange(len(pc2))
            np.random.shuffle(idxs)

        pc2 = pc2[idxs]
        correspondence_matrix = np.zeros((len(pc1), len(pc2)))
        correspondence_matrix[np.arange(len(pc1)), idxs] = 1
        correspondence_matrix = correspondence_matrix.T

        # make pc1, pc2 and correspondence matrix torch tensors float32
        pc1 = torch.from_numpy(pc1).float()
        pc2 = torch.from_numpy(pc2).float()
        correspondence_matrix = torch.from_numpy(correspondence_matrix).float()#.long()
        idx = torch.from_numpy(np.array([idx])).long()

        return correspondence_matrix, pc1, pc2, idx


class AcidDataModule(LightningDataModule):
    name = 'teddy'

    def __init__(
        self,
        data_dir: str,
        test_data_dir: str,
        val_split: float = 0.2,
        # test_split: float = 0.1,
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        object_count: int = 0,
        pair_count: int = 0,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.test_data_dir = test_data_dir if test_data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        root_dir = "/workspace/ACID/data_ndf2"
        category = ["teddy"]
        self.trainset = AcidPairDataset(root_dir, "train", category, transform=True, object_count = object_count, pair_count = pair_count, all_points=True)
        self.valset = AcidPairDataset(root_dir, "val", category, transform=False, object_count = object_count, pair_count = pair_count, all_points=True)

    def train_dataloader(self):
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            # num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader 

    # def test_dataloader(self):
    #     loader = DataLoader(
    #         self.testset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         drop_last=self.drop_last,
    #         pin_memory=self.pin_memory,
    #     )
    #     return loader

if __name__ == '__main__':
    print('ok')



