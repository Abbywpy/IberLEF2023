import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L

import pandas as pd
from loguru import logger

class SpanishTweetsDataModule(L.LightningDataModule):
    def __init__(self,
                 train_dataset_path="data/full_data/cleaned/train_clean_encoded.csv",
                 val_dataset_path="data/full_data/cleaned/val_clean_encoded.csv",
                 test_dataset_path="data/test_data/cleaned/cleaned_politicES_phase_2_test_public.csv",
                 num_workers=0,
                 batch_size=2):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = SpanishTweetsDataset(self.train_dataset_path)
        self.val_dataset = SpanishTweetsDataset(self.val_dataset_path)
        self.test_dataset = SpanishTweetsDataset(self.test_dataset_path)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=80,
            num_workers=2,
            shuffle=False,
        )

class SpanishTweetsDataset(data.Dataset):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.gender = self.df["gender"]
        self.profession = self.df["profession"]
        self.ideology_binary = self.df["ideology_binary"]
        self.ideology_multiclass = self.df["ideology_multiclass"]
        self.tweet = self.df["cleaned_tweet"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        results = {"gender": self.gender[idx],
                   "profession": self.profession[idx],
                   "ideology_binary": self.ideology_binary[idx],
                   "ideology_multiclass": self.ideology_multiclass[idx],
                   "tweet": self.tweet[idx]}
        return results
