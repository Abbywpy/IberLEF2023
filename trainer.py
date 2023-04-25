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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

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
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

pl.seed_everything(42, workers=True)  # for reproducibility


class SpanishTweetsCLF(pl.LightningModule):
    def __init__(self, hidden_size, dropout_rate, num_layers, lr, freeze_lang_model=True, clf="simple", bias=False):
        super().__init__()

        self.attr = ['ideology_multiclass']
        self.attr_size = [4]

        self.MariaRoberta = MariaRoberta()
        self.PolitiBeto = PolitiBeto()
        self.TwitterXLM = TwitterXLM()
        
        self.metrics = {}
        for attr in self.attr:
            self.metrics[f"{attr}_precision"] = torchmetrics.Precision(num_classes=self.attr_size[self.attr.index(attr)], average='macro', task="multiclass").to(DEVICE)
            self.metrics[f"{attr}_recall"] = torchmetrics.Recall(num_classes=self.attr_size[self.attr.index(attr)], average='macro', task="multiclass").to(DEVICE)
            self.metrics[f"{attr}_f1"] = torchmetrics.classification.MulticlassF1Score(num_classes=self.attr_size[self.attr.index(attr)], average='macro', task="multiclass").to(DEVICE)

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
        ret = {"device": DEVICE, **x}
        ret.update(self.MariaRoberta(**ret))
        ret.update(self.PolitiBeto(**ret))
        ret.update(self.TwitterXLM(**ret))

        ret["concated_embeds"] = concat_embeds(**ret)

        for attr in self.attr:
            ret.update(getattr(self, f'clf_{attr}')(**ret))

        return [ret[f"pred_{attr}"] for attr in self.attr]

    def training_step(self, batch, batch_idx):
        ret = {"device": DEVICE, **batch}
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

            final_metric = (f1 * precision * recall) / 3

            self.log(f"train_{attr}_loss", attr_loss)
            self.log(f"train_{attr}_acc", accuracy(
                ret[f"pred_{attr}"], ret[attr]))
            self.log(f"train_{attr}_precision", precision)
            self.log(f"train_{attr}_recall", recall)
            self.log(f"train_{attr}_f1", f1)
            self.log(f"train_{attr}_final_metric", final_metric)

        return loss

    def validation_step(self, batch, batch_idx):
        ret = {"device": DEVICE, **batch}
        ret.update(self.MariaRoberta(**ret))
        ret.update(self.PolitiBeto(**ret))
        ret.update(self.TwitterXLM(**ret))
        ret["concated_embeds"] = concat_embeds(**ret)

        for attr in self.attr:
            ret.update(getattr(self, f'clf_{attr}')(**ret))

        loss = 0
        total_f1 = 0
        for attr in self.attr:
            attr_loss = cross_entropy_loss(ret[f"pred_{attr}"], ret[attr])
            loss += attr_loss
            
            # Calculate and log precision, recall, and F1-score
            precision = self.metrics[f"{attr}_precision"](ret[f"pred_{attr}"], ret[attr])
            recall = self.metrics[f"{attr}_recall"](ret[f"pred_{attr}"], ret[attr])
            f1 = self.metrics[f"{attr}_f1"](ret[f"pred_{attr}"], ret[attr])

            final_metric = (f1 * precision * recall) / 3

            self.log(f"valid_{attr}_loss", attr_loss)
            self.log(f"valid_{attr}_acc", accuracy(ret[f"pred_{attr}"], ret[attr]))
            self.log(f"valid_{attr}_precision", precision)
            self.log(f"valid_{attr}_recall", recall)
            self.log(f"valid_{attr}_f1", f1)
            self.log(f"train_{attr}_final_metric", final_metric)

            total_f1 += f1

        average_f1 = total_f1 / len(self.attr)
        self.log("valid_average_f1", average_f1)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-05)
        return optimizer


def main(hparams):
    if hparams.clf == "simple":
        with open('best_hyperparams_ideology_multi.yaml') as f:
            default_config = yaml.load(f, Loader=yaml.FullLoader)

            batch_size = hparams.batch_size if hparams.batch_size else default_config["batch_size"]
            epochs = hparams.epochs if hparams.epochs else default_config["epochs"]

            if hparams.path_to_checkpoint:
                model = SpanishTweetsCLF.load_from_checkpoint(hparams.path_to_checkpoint)
            else:
                model = SpanishTweetsCLF(clf="simple",
                                         freeze_lang_model=True,
                                         lr=default_config["learning_rate"],
                                         dropout_rate=default_config["dropout"],
                                         hidden_size=default_config["hidden_size"],
                                         num_layers=default_config["num_layers"],
                                         bias=False)

    else:
        raise NotImplementedError
    
    if hparams.tiny_train:
        dm = SpanishTweetsDataModule(
            train_dataset_path="data/tiny_data/tiny_cleaned_encoded_train.csv", # path leads to *very* small subset of practise data
            val_dataset_path="data/tiny_data/tiny_cleaned_encoded_development.csv", # path leads to *very* small subset of practise data
            num_workers=hparams.num_workers,
            batch_size=batch_size)
        print("Using tiny train")

    elif hparams.practise_train:
        dm = SpanishTweetsDataModule(
            train_dataset_path="data/practise_data/cleaned/cleaned_encoded_development_train.csv", # path leads to  practise data
            val_dataset_path="data/practise_data/cleaned/cleaned_encoded_development_test.csv", # path leads to practise data
            num_workers=hparams.num_workers,
            batch_size=batch_size)

    else:
        dm = SpanishTweetsDataModule(train_dataset_path="data/full_data/cleaned/train_clean_encoded.csv",
                                     val_dataset_path="data/full_data/cleaned/val_clean_encoded.csv",
                                     num_workers=hparams.num_workers,
                                     batch_size=batch_size)
        print("Using full train")

    wandb_logger = WandbLogger(project="spanish-tweets")
    if hparams.path_to_checkpoint:
        trainer = L.Trainer(resume_from_checkpoint=hparams.path_to_checkpoint,
                            callbacks=[EarlyStopping(monitor="valid_average_f1", mode="max", patience=3),
                                       ModelCheckpoint(monitor="valid_average_f1", mode="max", save_top_k=3,
                                                       save_last=False, verbose=True)],
                            accelerator=hparams.accelerator,
                            devices=1,
                            logger=wandb_logger,
                            max_epochs=epochs)

    else:
        trainer = L.Trainer(callbacks=[EarlyStopping(monitor="valid_average_f1", mode="max", patience=3),
                                       ModelCheckpoint(monitor="valid_average_f1", mode="max", save_top_k=3, save_last=False, verbose=True)],
                            accelerator=hparams.accelerator,
                            devices=1,
                            logger=wandb_logger,
                            max_epochs=epochs)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", "-a", default="cpu", help="Change to GPU if you have one (default: cpu)")
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size for training (default: from yaml file)")
    parser.add_argument("--num_workers", "-n", type=int, help="Number of workers for dataloader (default: 2)")
    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs to train (default: from yaml file)")
    parser.add_argument("--clf", "-c", type=str, default="simple", help="Classifier to use (default: simple)")
    parser.add_argument("--tiny_train", "-tiny", action="store_true", help="Use tiny train dataset (default: False)")
    parser.add_argument("--practise_train", "-practise", action="store_true", help="Use tiny train dataset (default: False)")
    parser.add_argument("--path_to_checkpoint", "-cp", help="Path to checkpoint to load (default: None)")

    args = parser.parse_args()

    main(args)
