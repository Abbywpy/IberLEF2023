import itertools
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import lightning as L
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger


from models.lang_models.maria import MariaRoberta
from models.lang_models.politibeto import PolitiBeto
from models.lang_models.xlmt import TwitterXLM
from models.clfs.simpleCLF import SimpleCLF

from dataloader import SpanishTweetsDataModule

from models.utils import concat_embeds
from loss import cross_entropy_loss

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SpanishTweetsCLF(pl.LightningModule):
    def __init__(self, freeze_lang_model=True, lr=1e-3):
        super().__init__()
        self.MariaRoberta = MariaRoberta()
        self.PolitiBeto = PolitiBeto()
        self.TwitterXLM = TwitterXLM()
        # need to change this to a classifier
        self.clf = SimpleCLF(input_size=81264)
        self.lr = lr
        if freeze_lang_model:
            for param in self.MariaRoberta.parameters():
                param.requires_grad = False

            for param in self.PolitiBeto.parameters():
                param.requires_grad = False

            for param in self.TwitterXLM.parameters():
                param.requires_grad = False

    def forward(self, x):
        ret = {**x}
        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)
        ret |= self.TwitterXLM(**ret)

        ret["concated_embeds"] = concat_embeds(**ret)

        ret |= self.clf(**ret)

        return ret["result"]

    def training_step(self, batch, batch_idx):
        ret = {**batch}
        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)
        ret |= self.TwitterXLM(**ret)
        ret["concated_embeds"] = concat_embeds(**ret)

        ret |= self.clf(**ret)

        # TODO: add accuracy, f1, etc.
        loss = 0
        # TODO: Add other categories loss
        gender_loss = cross_entropy_loss(ret["pred_gender"], ret["gender"])
        self.log("train_gender_loss", gender_loss)
        loss += gender_loss
        return loss

    def validation_step(self, batch, batch_idx):
        ret = {**batch}
        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)
        ret |= self.TwitterXLM(**ret)
        ret["concated_embeds"] = concat_embeds(**ret)

        ret |= self.clf(**ret)

        # TODO: add accuracy, f1, etc.
        loss = 0
        # TODO: Add other categories loss
        gender_loss = cross_entropy_loss(ret["pred_gender"], ret["gender"])
        self.log("valid_gender_loss", gender_loss)
        loss += gender_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main(hparams):
    model = SpanishTweetsCLF()

    # TODO: add train_dataset_path and val_dataset_path
    dm = SpanishTweetsDataModule(train_dataset_path="data/practise_data/cleaned/cleaned_development_train.csv",
                                 val_dataset_path="data/practise_data/cleaned/cleaned_development_test.csv", batch_size=hparams.batch_size)
    wandb_logger = WandbLogger(project="spanish-tweets")
    trainer = L.Trainer(accelerator=hparams.accelerator, logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", "-a", default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=None)
    args = parser.parse_args()

    main(args)
