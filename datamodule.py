import os
import torch
import json

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F

from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime

import numpy as np


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/",
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_path = data_path

        self.val_transform = T.Compose(
            [
                T.Resize(224),
                T.ToTensor(),
                self.normalize,
            ]
        )

        self.train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, padding=4),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.Resize(224),
                T.ToTensor(),
                self.normalize,
            ]
        )
        self.args = kwargs

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.normalize = T.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
        )

    @property
    def num_classes(self):
        return 10  # len(self.train_dataset.classes)

    @property
    def classes(self):
        return self.data_train.classes

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage=None):
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`, `self.test_dataset`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        data_path = self.data_path

        train_data_dir = data_path + "/train"
        val_data_dir = data_path + "/val"
        test_data_dir = data_path + "/test"

        train_count = self.get_num_files(train_data_dir)
        val_count = self.get_num_files(val_data_dir)
        test_count = self.get_num_files(test_data_dir)

        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            trainset = ImageFolder(self.train_data_dir, transform=self.train_transform)
            valset = ImageFolder(self.val_data_dir, transform=self.val_transform)
            testset = ImageFolder(self.test_data_dir, transform=self.val_transform)

            self.train_dataset, self.val_dataset, self.test_dataset = (
                trainset,
                valset,
                testset,
            )

    def train_dataloader(self):
        self.train_data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.get("train_batch_size", self.hparams.batch_size),
            num_workers=self.args.get("train_num_workers", self.hparams.num_workers),
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        return self.train_data_loader

    def val_dataloader(self):
        self.val_data_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.args.get("val_batch_size", self.hparams.batch_size),
            num_workers=self.args.get("val_num_workers", self.hparams.num_workers),
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        return self.val_data_loader

    def test_dataloader(self):
        self.test_data_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.get("val_batch_size", self.hparams.batch_size),
            num_workers=self.args.get("val_num_workers", self.hparams.num_workers),
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        return self.test_data_loader

    def teardown(self, stage=None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict):
        """Things to do when loading checkpoint."""
        pass
