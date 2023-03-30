import torch


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def concat_embeds(maria_embed, politibeto_embed, **kwargs):
    # TODO: need to change become flexiable to the batch size
    return torch.cat([maria_embed, politibeto_embed], dim=1)
