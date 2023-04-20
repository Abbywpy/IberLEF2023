from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn

MODEL_NAME = "nlp-cimat/politibeto"


class PolitiBeto(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(PolitiBeto, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_sentence_embeddings(self, embeddings):
        # Calculate the mean of the token embeddings along the sequence dimension (dim=1)
        sentence_embeddings = embeddings.mean(dim=1)
        return sentence_embeddings

    def normalize_embeddings(self, embeddings):
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings.div(norms)
        return normalized_embeddings

    def forward(self, tweet, device, **kwargs):

        encoded_input = self.tokenizer(
            tweet, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.get_sentence_embeddings(model_output[0])
        normalized_embeddings = self.normalize_embeddings(sentence_embeddings)
        return {"politibeto_embed": normalized_embeddings, **kwargs}
    