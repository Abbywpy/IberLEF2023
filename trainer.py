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

from models.utils import concat_embeds
from loss import cross_entropy_loss


class SpanishTweetsCLF(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.MariaRoberta = MariaRoberta()
        self.PolitiBeto = PolitiBeto()
        self.clf = nn.Linear(768, 2)  # need to change this to a classifier
        self.lr = lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        ret = {"data": x}
        ret |= self.MariaRoberta(**ret)
        ret |= self.MariaRoberta(**ret)

        ret["concated_embeds"] = concat_embeds(**ret)

        ret |= self.clf(**ret)

        return ret["result"]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        data, gt = batch

        ret = {"data": data}
        ret |= self.MariaRoberta(**ret)
        ret |= self.MariaRoberta(**ret)

        # TODO: add accuracy, f1, etc.
        loss = cross_entropy_loss(ret["result"], gt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the valid loop. It is independent of forward
        data, gt = batch

        ret = {"data": data}
        ret |= self.MariaRoberta(**ret)
        ret |= self.MariaRoberta(**ret)

        # TODO: add accuracy, f1, etc.
        loss = cross_entropy_loss(ret["result"], gt)
        self.log("vaild_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
