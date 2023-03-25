import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import lightning as L

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from models.lang_models.maria import MariaRoberta
from models.lang_models.politibeto import PolitiBeto
from models.clfs.simpleCLF import SimpleCLF

from dataloader import SpanishTweetsDataModule

from models.utils import concat_embeds
from loss import cross_entropy_loss


class SpanishTweetsCLF(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.MariaRoberta = MariaRoberta()
        self.PolitiBeto = PolitiBeto()
        self.clf = SimpleCLF()  # need to change this to a classifier
        self.lr = lr

    def forward(self, x):
        ret = {**x}
        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)

        ret["concated_embeds"] = concat_embeds(**ret)

        ret |= self.clf(**ret)

        return ret["result"]

    def training_step(self, batch, batch_idx):
        ret = {**batch}
        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)
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
