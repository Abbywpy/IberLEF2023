from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import mean_pooling


MODEL_NAME = 'cardiffnlp/twitter-xlm-roberta-base'


class TwitterXLM(nn.Module):
    """
    This class is a wrapper for the Twitter-XLM-Roberta-base model from HuggingFace
    """

    def __init__(self, model_name=MODEL_NAME):
        super(TwitterXLM, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def forward(self, tweet, **kwargs):
        """
        Return the embedding part of the model

        :param
        tweet: a string with the sentence to be embedded
        """

        encoded_input = self.tokenizer(
            tweet, padding=True, truncation=True, max_length=128, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(
                model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return {"xlmt_embed": sentence_embeddings, **kwargs}
