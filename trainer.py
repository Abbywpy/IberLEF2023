import os
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
from models.clfs.mlclCLF import mlclCLF

from dataloader import SpanishTweetsDataModule

from models.utils import concat_embeds
from models.clfs.losses import cross_entropy_loss, accuracy

from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'medianAUC',
           'meanAUPR', 'medianAUPR', 'meanFDR', 'medianFDR', 'p_at_1', 'p_at_3', 'p_at_5']
CLF_DICT = {"simple": {'gender': 2,
                       'profession': 3,
                       'ideology_binary': 2, 'ideology_multiclass': 4},
            "mlcl": {"mlcl": 24}
            }


class SpanishTweetsCLF(pl.LightningModule):
    def __init__(self,
                 freeze_lang_model=True,
                 clf_type="simple",
                 lr=1e-3):

        super().__init__()

        self.MariaRoberta = MariaRoberta()
        self.PolitiBeto = PolitiBeto()
        self.TwitterXLM = TwitterXLM()

        # classifiers
        self.clf_type = clf_type
        self.clf_attr = CLF_DICT[clf_type]

        logger.info(self.clf_attr)

        # TODO: fineturn SimpleCLF classifier
        if clf_type == "simple":
            logger.info("running simple classifier")
            for attr, s in self.clf_attr.items(): 
                setattr(self, f"clf_{attr}", SimpleCLF(
                    attr_name=attr, output_size=s))

            # TODO: fineturn mlclCLF classifier
        elif clf_type == "mlcl":
            logger.info("running mlcl classifier")
            self.clf_mlcl = mlclCLF(latent_dim=1000)

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

        for attr in self.clf_attr:
            ret = {**ret, "y": ret[attr]}
            ret |= getattr(self, f'clf_{attr}')(**ret)

        return [ret[f"pred_{attr}"] for attr in self.clf_attr]

    def training_step(self, batch, batch_idx):
        ret = {**batch}
        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)
        ret |= self.TwitterXLM(**ret)
        ret["concated_embeds"] = concat_embeds(**ret)

        for attr in self.clf_attr:
            ret = {**ret, "y": ret[attr]}
            ret |= getattr(self, f'clf_{attr}')(**ret)

        return ret["total_loss"]

    def validation_step(self, batch, batch_idx):
        ret = {**batch}

        ret |= self.MariaRoberta(**ret)
        ret |= self.PolitiBeto(**ret)
        ret |= self.TwitterXLM(**ret)
        ret["concated_embeds"] = concat_embeds(**ret)
            
        logger.info(self.clf_attr)
        logger.info(ret)
        for attr in self.clf_attr:
            ret = {**ret, "y": ret[attr]}
            ret |= getattr(self, f'clf_{attr}')(**ret)

        return ret["total_loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main(hparams):
    model = SpanishTweetsCLF(clf_type = hparams.clf_type)

    # TODO: change train_dataset_path and val_dataset_path
    dm = SpanishTweetsDataModule(train_dataset_path="data/practise_data/cleaned/cleaned_encoded_development_train.csv",
                                 val_dataset_path="data/practise_data/cleaned/cleaned_encoded_development_test.csv", batch_size=hparams.batch_size)
    wandb_logger = WandbLogger(project="spanish-tweets")
    trainer = L.Trainer(accelerator=hparams.accelerator, logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", "-a", default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=None)
    parser.add_argument("--clf-type", "-clf", type=str, default=None)
    args = parser.parse_args()

    main(args)
