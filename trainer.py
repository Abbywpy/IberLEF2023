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
from loss import cross_entropy_loss, accuracy

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SpanishTweetsCLF(pl.LightningModule):
    def __init__(self, freeze_lang_model=True, clf="simple", lr=1e-3):
        super().__init__()

        self.attr = ['gender', 'profession',
                     'ideology_binary', 'ideology_multiclass']
        self.attr_size = [2, 3, 2, 4]

        self.MariaRoberta = MariaRoberta()
        self.PolitiBeto = PolitiBeto()
        self.TwitterXLM = TwitterXLM()

        # TODO: fineturn SimpleCLF classifier
        if clf == "simple":
            for attr, s in zip(self.attr, self.attr_size):
                setattr(self, f"clf_{attr}", SimpleCLF(
                    attr_name=attr, output_size=s))

        # TODO: add cl classifier
        else:
            raise NotImplementedError

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

        for attr in self.attr:
            ret |= getattr(self, f'clf_{attr}')(**ret)

        return [ret[f"pred_{attr}"] for attr in self.attr]

    def training_step(self, batch, batch_idx):
        ret = {**batch}
        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)
        ret |= self.TwitterXLM(**ret)
        ret["concated_embeds"] = concat_embeds(**ret)

        for attr in self.attr:
            ret |= getattr(self, f'clf_{attr}')(**ret)

        # TODO: add f1, etc.
        loss = 0
        for attr in self.attr:
            attr_loss = cross_entropy_loss(ret[f"pred_{attr}"], ret[attr])

            self.log(f"train_{attr}_loss", attr_loss)
            self.log(f"train_{attr}_acc", accuracy(
                ret[f"pred_{attr}"], ret[attr]))

            loss += attr_loss
        return loss

    def validation_step(self, batch, batch_idx):
        ret = {**batch}
        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)
        ret |= self.TwitterXLM(**ret)
        ret["concated_embeds"] = concat_embeds(**ret)

        for attr in self.attr:
            ret |= getattr(self, f'clf_{attr}')(**ret)

        # TODO: add f1, etc.
        loss = 0
        for attr in self.attr:
            attr_loss = cross_entropy_loss(ret[f"pred_{attr}"], ret[attr])

            self.log(f"valid_{attr}_loss", attr_loss)
            self.log(f"valid_{attr}_acc", accuracy(
                ret[f"pred_{attr}"], ret[attr]))

            loss += attr_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main(hparams):
    model = SpanishTweetsCLF()

    # TODO: change train_dataset_path and val_dataset_path
    if hparams.tiny_train == "yes":
        dm = SpanishTweetsDataModule(train_dataset_path="data/practise_data/cleaned/cleaned_encoded_development_train.csv",
                                     val_dataset_path="data/practise_data/cleaned/cleaned_encoded_development_test.csv", batch_size=hparams.batch_size)
        print("Using tiny train")
    else:
        dm = SpanishTweetsDataModule(train_dataset_path="data/full_data/cleaned/train_clean_encoded.csv",
                                     val_dataset_path="data/full_data/cleaned/val_clean_encoded.csv", batch_size=hparams.batch_size)
        print("Using full train")
    
    wandb_logger = WandbLogger(project="spanish-tweets")
    trainer = L.Trainer(accelerator=hparams.accelerator, logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", "-a", default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=None)
    parser.add_argument("--tiny-train", "-tiny", type=str, default="yes")
    args = parser.parse_args()

    main(args)
