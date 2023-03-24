from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mean_pooling


MODEL_NAME = 'PlanTL-GOB-ES/roberta-base-bne'


class MariaRoberta(nn.Module):
    """
    This class is a wrapper for the MariaRoberta model from HuggingFace
    """

    def __init__(self, MODEL_NAME):
        super(MariaRoberta, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    def forward(self, sentence):
        """
        Return the embedding part of the model

        :param
        sentence: a string with the sentence to be embedded
        """

        encoded_input = self.tokenizer(
            sentence, padding=True, truncation=True, max_length=128, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(
                model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
