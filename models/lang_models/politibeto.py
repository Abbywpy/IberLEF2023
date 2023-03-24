from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch
import torch.nn as nn

MODEL_NAME = "nlp-cimat/politibeto"


class PolitiBeto(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(PolitiBeto, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def get_embeddings(self, text_batch):
        inputs = self.tokenizer(
            text_batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
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

    def forward(self, data, **kwargs):
        token_embeddings = self.get_embeddings(data)
        sentence_embeddings = self.get_sentence_embeddings(token_embeddings)
        normalized_embeddings = self.normalize_embeddings(sentence_embeddings)
        return {"politibeto_embed": normalized_embeddings, **kwargs}
    