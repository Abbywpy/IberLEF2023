import os
import yaml
from argparse import ArgumentParser
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchmetrics

import wandb
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
from utils import DictAsMember


os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

CLF_DICT = {"simple": [
    'gender',
    'profession',
    'ideology_binary',
    'ideology_multiclass'],
    "mlcl": ["mlcl"]
}

pl.seed_everything(42, workers=True)  # for reproducibility


class SpanishTweetsCLF(pl.LightningModule):
    def __init__(self, freeze_lang_model=True, clf_type="simple", hp_path="./clf_hp", bias=False, lr=2.0e-5):
        super().__init__()

        self.MariaRoberta = MariaRoberta()
        self.PolitiBeto = PolitiBeto()
        self.TwitterXLM = TwitterXLM()
        # classifiers
        self.clf_type = clf_type
        self.clf_attr = CLF_DICT[clf_type]
        self.lr = lr

        # TODO: finetune SimpleCLF classifier
        if clf_type == "simple":
            logger.info("running simple classifier")

            for attr in self.clf_attr:
                attr_hp_path = os.path.join(hp_path, f"{attr}.yaml")

                with open(attr_hp_path) as f:
                    attr_hp = DictAsMember(yaml.safe_load(f))
                    logger.info(attr_hp)
                    setattr(self, f"{attr}_hp", attr_hp)
                setattr(self, f"{attr}_precision", torchmetrics.Precision(
                    num_classes=attr_hp.output_size, average='macro', task="multiclass").to(DEVICE))
                setattr(self, f"{attr}_recall", torchmetrics.Recall(
                    num_classes=attr_hp.output_size, average='macro', task="multiclass").to(DEVICE))
                setattr(self, f"{attr}_f1", torchmetrics.classification.MulticlassF1Score(
                    num_classes=attr_hp.output_size, average='macro', task="multiclass").to(DEVICE))

                setattr(self, f"clf_{attr}", SimpleCLF(
                    attr_name=attr,
                    output_size=attr_hp.output_size,
                    bias=bias,
                    dropout_rate=attr_hp.dropout,
                    hidden_size=attr_hp.hidden_size,
                    num_layers=attr_hp.num_layers))

        # TODO: add cl classifier and config file for it
        else:
            raise NotImplementedError

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

        for attr in self.clf_attr:
            ret.update(getattr(self, f'clf_{attr}')(**ret))

        return [ret[f"pred_{attr}"] for attr in self.clf_attr]

    def training_step(self, batch, batch_idx):
        ret = {"device": DEVICE, **batch}
        ret.update(self.MariaRoberta(**ret))
        ret.update(self.PolitiBeto(**ret))
        ret.update(self.TwitterXLM(**ret))
        ret["concated_embeds"] = concat_embeds(**ret)

        loss = 0
        wandb_logger = {}
        for attr in self.clf_attr:
            ret.update(getattr(self, f'clf_{attr}')(**ret))

            attr_loss = cross_entropy_loss(ret[f"pred_{attr}"], ret[attr])
            loss += attr_loss

            # Calculate and log precision, recall, and F1-score
            precision = getattr(self, f'{attr}_precision')(
                ret[f"pred_{attr}"], ret[attr])
            recall = getattr(self, f'{attr}_recall')(
                ret[f"pred_{attr}"], ret[attr])
            f1 = getattr(self, f'{attr}_f1')(ret[f"pred_{attr}"], ret[attr])
            final_metric = (f1 * precision * recall) / 3


            wandb_logger[f"train/{attr}/loss"] = attr_loss
            wandb_logger[f"train/{attr}/acc"] = accuracy(
                ret[f"pred_{attr}"], ret[attr])
            wandb_logger[f"train/{attr}/precision"] = precision
            wandb_logger[f"train/{attr}/recall"] = recall
            wandb_logger[f"train/{attr}/f1"] = f1
            wandb_logger[f"train/{attr}/final_metric"] = final_metric
            if self.global_step % 80 == 0:
                cm = wandb.plot.confusion_matrix(
                    y_true=ret[attr].tolist(),
                    preds=ret[f"pred_{attr}"].argmax(dim=1).tolist(),
                    class_names=getattr(self, f'{attr}_hp').class_name)
                wandb_logger[f"train/{attr}/conf_mat"] = cm
        self.logger.experiment.log(wandb_logger)


        return loss

    def validation_step(self, batch, batch_idx):
        ret = {"device": DEVICE, **batch}
        ret.update(self.MariaRoberta(**ret))
        ret.update(self.PolitiBeto(**ret))
        ret.update(self.TwitterXLM(**ret))
        ret["concated_embeds"] = concat_embeds(**ret)

        loss = 0
        total_f1 = 0
        total_final_metric = 0
        wandb_logger = {}
        for attr in self.clf_attr:
            ret.update(getattr(self, f'clf_{attr}')(**ret))

            attr_loss = cross_entropy_loss(ret[f"pred_{attr}"], ret[attr])
            loss += attr_loss

            # Calculate and log precision, recall, and F1-score
            precision = getattr(self, f'{attr}_precision')(
                ret[f"pred_{attr}"], ret[attr])
            recall = getattr(self, f'{attr}_recall')(
                ret[f"pred_{attr}"], ret[attr])
            f1 = getattr(self, f'{attr}_f1')(ret[f"pred_{attr}"], ret[attr])
            final_metric = (f1 * precision * recall) / 3
            total_final_metric += final_metric
            total_f1 += f1

            
            wandb_logger[f"valid/{attr}/loss"] = attr_loss
            wandb_logger[f"valid/{attr}/acc"] = accuracy(
                ret[f"pred_{attr}"], ret[attr])
            wandb_logger[f"valid/{attr}/precision"] = precision
            wandb_logger[f"valid/{attr}/recall"] = recall
            wandb_logger[f"valid/{attr}/f1"] = f1
            wandb_logger[f"valid/{attr}/final_metric"] = final_metric
            self.log(f"valid/{attr}/loss", attr_loss)
            self.log(f"valid/{attr}/acc", accuracy(
                ret[f"pred_{attr}"], ret[attr]))
            self.log(f"valid/{attr}/precision", precision)
            self.log(f"valid/{attr}/recall", recall)
            self.log(f"valid/{attr}/f1", f1)
            self.log(f"valid/{attr}/final_metric", final_metric)
            if self.global_step % 80 == 0:
                    cm = wandb.plot.confusion_matrix(
                        y_true=ret[attr].tolist(),
                        preds=ret[f"pred_{attr}"].argmax(dim=1).tolist(),
                        class_names=getattr(self, f'{attr}_hp').class_name)
                    wandb_logger[f"valid/{attr}/conf_mat"] = cm

        average_f1 = total_f1 / len(self.clf_attr)
        wandb_logger["valid/average_f1"] = average_f1
        self.log("valid/average_f1", average_f1)
        avg_final_metric = total_final_metric / len(self.clf_attr)
        wandb_logger["valid/average_final_metric"] = avg_final_metric
        self.log("valid/average_final_metric", avg_final_metric)
        self.logger.experiment.log(wandb_logger)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main(hparams):
    if hparams.clf == "simple":
        model = SpanishTweetsCLF(clf_type="simple",
                                 freeze_lang_model=hparams.freeze,
                                 lr=hparams.learning_rate,
                                 bias=False)

    else:
        raise NotImplementedError

    if hparams.tiny_train:
        # path leads to *very* small subset of practise data
        train_dataset_path = "data/tiny_data/tiny_cleaned_encoded_train.csv"
        val_dataset_path = "data/tiny_data/tiny_cleaned_encoded_development.csv"
        logger.info("Using tiny train")

    elif hparams.practise_train:
        # path leads to practise data
        train_dataset_path = "data/practise_data/cleaned/cleaned_encoded_development_train.csv"
        val_dataset_path = "data/practise_data/cleaned/cleaned_encoded_development_test.csv"
        logger.info("Using practise train")
    else:
        train_dataset_path = "data/full_data/cleaned/train_clean_encoded.csv"
        val_dataset_path = "data/full_data/cleaned/val_clean_encoded.csv"
        logger.info("Using full train")

    dm = SpanishTweetsDataModule(train_dataset_path,
                                 val_dataset_path,
                                 num_workers=hparams.num_workers,
                                 batch_size=hparams.batch_size)
    wandb_logger = WandbLogger(project="spanish-tweets", name=hparams.run_name)

    trainer = L.Trainer(callbacks=[
                        EarlyStopping(monitor="valid/average_final_metric", mode="max", patience=3),
                        ModelCheckpoint(monitor="valid/average_final_metric", mode="max", save_top_k=3, save_last=False, verbose=True, filename="{epoch}-{valid/average_final_metric:.2f}"),
                        ModelCheckpoint(monitor="valid/profession/final_metric", mode="max", save_top_k=1, save_last=False, verbose=True, filename="{epoch}-{valid/profession/final_metric:.2f}"),
                        ModelCheckpoint(monitor="valid/gender/final_metric", mode="max", save_top_k=1, save_last=False, verbose=True, filename="{epoch}-{valid/gender/final_metric:.2f}"),
                        ModelCheckpoint(monitor="valid/ideology_multiclass/final_metric", mode="max", save_top_k=1, save_last=False, verbose=True, filename="{epoch}-{valid/ideology_multiclass/final_metric:.2f}"),
                        ModelCheckpoint(monitor="valid/ideology_binary/final_metric", mode="max", save_top_k=1, save_last=False, verbose=True, filename="{epoch}-{valid/ideology_binary/final_metric:.2f}"),],
                        accelerator=hparams.accelerator,
                        devices=1,
                        logger=wandb_logger,
                        max_epochs=hparams.epochs)

    trainer.fit(model, dm, ckpt_path=hparams.path_to_checkpoint)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", "-a", default="cpu",
                        help="Change to GPU if you have one (default: cpu)")
    parser.add_argument("--batch-size", "-b", default=8, type=int,
                        help="Batch size for training (default: from yaml file)")
    parser.add_argument("--num_workers", "-w", type=int,
                        help="Number of workers for dataloader (default: 2)")
    parser.add_argument("--epochs", "-e", default=15, type=int,
                        help="Number of epochs to train (default: from yaml file)")
    parser.add_argument("--learning-rate", "-lr",
                        default=2.0e-5, type=float, help="learning-rate")

    parser.add_argument("--run-name", "-n", default=None,
                        type=str, help="learning-rate")

    parser.add_argument("--clf", "-c", type=str, default="simple",
                        help="Classifier to use (default: simple)")
    parser.add_argument("--hp_path", "-hp", type=str,
                        default="./clf_hp", help="hp files for classifiers")
    parser.add_argument("--tiny_train", "-tiny", action="store_true",
                        help="Use tiny train dataset (default: False)")
    parser.add_argument("--practise_train", "-practise", action="store_true",
                        help="Use tiny train dataset (default: False)")
    parser.add_argument("--path_to_checkpoint", "-cp",
                        help="Path to checkpoint to load (default: None)")
    parser.add_argument("--freeze", default=True, type=bool, help="Freeze language model (default: True)")

    args = parser.parse_args()

    main(args)
