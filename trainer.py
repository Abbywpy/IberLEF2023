import itertools
from argparse import ArgumentParser

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchmetrics

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

from loguru import logger

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

pl.seed_everything(42, workers=True)  # for reproducibility


class SpanishTweetsCLF(pl.LightningModule):
    def __init__(self, freeze_lang_model=True, clf="simple", bias=False, dropout_rate=0.15, hidden_size=256, num_layers=2, lr=1e-3):
        super().__init__()

        self.attr = ['gender', 'profession',
                     'ideology_binary', 'ideology_multiclass']
        self.attr_size = [2, 3, 2, 4]

        self.MariaRoberta = MariaRoberta()
        self.PolitiBeto = PolitiBeto()
        self.TwitterXLM = TwitterXLM()
        
        # Add torchmetrics instances for precision, recall, and F1-score
        self.metrics = {}
        for attr in self.attr:
            self.metrics[f"{attr}_precision"] = torchmetrics.Precision(num_classes=self.attr_size[self.attr.index(attr)], average='macro', task="multiclass")
            self.metrics[f"{attr}_recall"] = torchmetrics.Recall(num_classes=self.attr_size[self.attr.index(attr)], average='macro', task="multiclass")
            self.metrics[f"{attr}_f1"] = torchmetrics.classification.MulticlassF1Score(num_classes=self.attr_size[self.attr.index(attr)], average='macro', task="multiclass")

        # TODO: finetune SimpleCLF classifier
        if clf == "simple":
            for attr, s in zip(self.attr, self.attr_size):
                setattr(self, f"clf_{attr}", SimpleCLF(
                    attr_name=attr, output_size=s, bias=bias, dropout_rate=dropout_rate, hidden_size=hidden_size, num_layers=num_layers))

        # TODO: add cl classifier and config file for it
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
        logger.info([f"{k}_{v.device}" for k, v in ret.items()])
        ret.update(self.MariaRoberta(**ret))
        ret.update(self.PolitiBeto(**ret))
        ret.update(self.TwitterXLM(**ret))

        ret["concated_embeds"] = concat_embeds(**ret)
        

        for attr in self.attr:
            ret.update(getattr(self, f'clf_{attr}')(**ret))

        return [ret[f"pred_{attr}"] for attr in self.attr]

    def training_step(self, batch, batch_idx):
        ret = {**batch}
        logger.info([f"{k}_{v.device}" for k, v in ret.items()])
        ret.update(self.MariaRoberta(**ret))
        ret.update(self.PolitiBeto(**ret))
        ret.update(self.TwitterXLM(**ret))
        ret["concated_embeds"] = concat_embeds(**ret)

        for attr in self.attr:
            ret.update(getattr(self, f'clf_{attr}')(**ret))

        loss = 0
        for attr in self.attr:
            attr_loss = cross_entropy_loss(ret[f"pred_{attr}"], ret[attr])
            loss += attr_loss
            
            # Calculate and log precision, recall, and F1-score
            precision = self.metrics[f"{attr}_precision"](ret[f"pred_{attr}"], ret[attr])
            recall = self.metrics[f"{attr}_recall"](ret[f"pred_{attr}"], ret[attr])
            f1 = self.metrics[f"{attr}_f1"](ret[f"pred_{attr}"], ret[attr])

            self.log(f"train_{attr}_loss", attr_loss)
            self.log(f"train_{attr}_acc", accuracy(
                ret[f"pred_{attr}"], ret[attr]))
            self.log(f"train_{attr}_precision", precision)
            self.log(f"train_{attr}_recall", recall)
            self.log(f"train_{attr}_f1", f1)

        return loss

    def validation_step(self, batch, batch_idx):
        ret = {**batch}
        ret.update(self.MariaRoberta(**ret))
        ret.update(self.PolitiBeto(**ret))
        ret.update(self.TwitterXLM(**ret))
        ret["concated_embeds"] = concat_embeds(**ret)

        for attr in self.attr:
            ret.update(getattr(self, f'clf_{attr}')(**ret))

        loss = 0
        for attr in self.attr:
            attr_loss = cross_entropy_loss(ret[f"pred_{attr}"], ret[attr])
            loss += attr_loss
            
            # Calculate and log precision, recall, and F1-score
            precision = self.metrics[f"{attr}_precision"](ret[f"pred_{attr}"], ret[attr])
            recall = self.metrics[f"{attr}_recall"](ret[f"pred_{attr}"], ret[attr])
            f1 = self.metrics[f"{attr}_f1"](ret[f"pred_{attr}"], ret[attr])
            
            self.log(f"valid_{attr}_loss", attr_loss)
            self.log(f"valid_{attr}_acc", accuracy(
                ret[f"pred_{attr}"], ret[attr]))
            self.log(f"valid_{attr}_precision", precision)
            self.log(f"valid_{attr}_recall", recall)
            self.log(f"valid_{attr}_f1", f1)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main(hparams):
    if hparams.clf == "simple":
        with open('best_hyperparams_simpleCLF.yaml') as f:
            default_config = yaml.load(f, Loader=yaml.FullLoader)
            
            # TODO: after hparam search remove this model and uncomment the model, epochs and batch_size below
            model = SpanishTweetsCLF(clf="simple", freeze_lang_model=True, lr=1e-3, dropout_rate=0.2, hidden_size=128, num_layers=2, bias=False)

            
            #model = SpanishTweetsCLF(clf="simple", freeze_lang_model=True, lr=default_config["lr", dropout_rate=default_config["dropout"], hidden_size=default_config["hidden_size"], num_layers=default_config["num_layers"], bias=False)
            
            #epochs = default_config["epochs"]
            #batch_size = default_config["batch_size"]
    else:
        raise NotImplementedError
    
    # TODO: after hparam search the batch_size from default_config can be used
    if hparams.tiny_train:
        dm = SpanishTweetsDataModule(
            train_dataset_path="data/tiny_data/tiny_cleaned_encoded_train.csv", # path leads to *very* small subset of practise data
            val_dataset_path="data/tiny_data/tiny_cleaned_encoded_development.csv", # path leads to *very* small subset of practise data
            batch_size=hparams.batch_size)
        print("Using tiny train")
    elif hparams.practise:
        dm = SpanishTweetsDataModule(
            train_dataset_path="data/practise_data/cleaned/cleaned_encoded_development_train.csv", # path leads to  practise data
            val_dataset_path="data/practise_data/cleaned/cleaned_encoded_development_test.csv", # path leads to practise data
            batch_size=hparams.batch_size)
    else:
        dm = SpanishTweetsDataModule(train_dataset_path="data/full_data/cleaned/train_clean_encoded.csv",
                                     val_dataset_path="data/full_data/cleaned/val_clean_encoded.csv",
                                     num_workers=hparams.num_workers,
                                     batch_size=hparams.batch_size)
        print("Using full train")

    wandb_logger = WandbLogger(project="spanish-tweets")
    # TODO: after hparam search the epochs from default_config can be used
    trainer = L.Trainer(accelerator=hparams.accelerator, devices=1, logger=wandb_logger, max_epochs=hparams.epochs)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", "-a", default="cpu", help="Change to GPU if you have one (default: cpu)")
    parser.add_argument("--batch-size", "-b", type=int, default=2, help="Batch size for training (default: 2)")
    parser.add_argument("--num_workers", "-n", type=int, default=2, help="Number of workers for dataloader (default: 2)")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs to train (default: 3)")
    parser.add_argument("--clf", "-c", type=str, default="simple", help="Classifier to use (default: simple)")
    parser.add_argument("--tiny_train", "-tiny", action="store_true", help="Use tiny train dataset (default: False)")
    parser.add_argument("--practise_train", "-practise", action="store_true", help="Use tiny train dataset (default: False)")
    args = parser.parse_args()

    main(args)
