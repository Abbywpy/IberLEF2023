# write a simple multi-layer binary classifier
#  - input: 768
#  - output: 2
#  - hidden layers: 2
#  - hidden layer size: 256
#  - activation: ReLU
#  - loss: CrossEntropyLoss
#  - optimizer: Adam
#  - learning rate: 1e-3
#  - batch size: 32
#  - epochs: 10

# Path: models/clfs/simpleCLF.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCLF(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=2, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, concated_embeds, **kwargs):
        concated_embeds = F.relu(self.fc1(concated_embeds))
        result = self.fc2(concated_embeds)
        return {"result": result, **kwargs}
