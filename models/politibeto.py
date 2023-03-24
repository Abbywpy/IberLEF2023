import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch
import torch.nn as nn


class PolitiBeto(nn.Module):
    def __init__(self):
        super(PolitiBeto, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("nlp-cimat/politibeto")
        self.model = AutoModelForMaskedLM.from_pretrained("nlp-cimat/politibeto")

    def get_embeddings(self, text_batch):
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs[0]

    def get_sentence_embeddings(self, embeddings):
        # Calculate the mean of the token embeddings along the sequence dimension (dim=1)
        sentence_embeddings = embeddings.mean(dim=1)
        return sentence_embeddings

    def normalize_embeddings(self, embeddings):
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings.div(norms)
        return normalized_embeddings

    def forward(self, data):

        embeddings_list = []

        if data.endswith(".csv"):
            df = pd.read_csv(data)
            text_data = df["cleaned_tweet"]
            batch_size = 32

        elif type(data) == str:
            text_data = [data]
            batch_size = 1

        for i in range(0, len(text_data), batch_size):
            text_batch = text_data[i:i + batch_size].tolist() if batch_size > 1 else text_data[i]
            token_embeddings_batch = self.get_embeddings(text_batch)
            sentence_embeddings_batch = self.get_sentence_embeddings(token_embeddings_batch)
            normalized_embeddings_batch = self.normalize_embeddings(sentence_embeddings_batch)
            embeddings_list.append(normalized_embeddings_batch)

        if len(embeddings_list) == 1:
            return embeddings_list[0]
        else:
            return embeddings_list

