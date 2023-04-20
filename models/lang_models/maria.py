from transformers import AutoModel
from transformers import AutoTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from models.utils import mean_pooling


MODEL_NAME = 'PlanTL-GOB-ES/roberta-base-bne'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
logger.error(device)

class MariaRoberta(nn.Module):
    """
    This class is a wrapper for the MariaRoberta model from HuggingFace
    """

    def __init__(self, model_name=MODEL_NAME):
        super(MariaRoberta, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, tweet, **kwargs):
        """
        Return the embedding part of the model

        :param
        tweet: a string with the sentence to be embedded
        """
        logger.info(tweet.device)
        encoded_input = self.tokenizer(
            tweet, padding=True, truncation=True, max_length=128, return_tensors='pt')

        with torch.no_grad():
            logger.info([f"{k}_{v.device}" for k, v in encoded_input.items()])
            model_output = self.model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(
                model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return {"maria_embed": sentence_embeddings, **kwargs}
