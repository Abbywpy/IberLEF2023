# Simple classifier for each attribute

# Path: models/clfs/simpleCLF.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCLF(nn.Module):
    def __init__(self, attr_name, hidden_size=256, output_size=2, num_layers=2, bias=False, dropout_rate=0.15):
        super().__init__()
        self.attr_name = attr_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers


        self.clf = nn.Sequential()
        for i in range(num_layers):
            self.clf.add_module(f"fc{i}", nn.LazyLinear(hidden_size, bias))
            self.clf.add_module(f"bn{i}", nn.BatchNorm1d(hidden_size))
            self.clf.add_module(f"relu{i}", nn.ReLU())
            self.clf.add_module(f"dropout{i}", nn.Dropout(dropout_rate))
        self.clf.add_module(f"lastlayer", nn.Linear(hidden_size, output_size))

    def forward(self, concated_embeds, **kwargs):
        pred = self.clf(concated_embeds)
        return {f"pred_{self.attr_name}": pred, **kwargs}
