import torch
from dataloader import SpanishTweetsDataModule
from transformers import AutoTokenizer
from argparse import Namespace
from trainer import SpanishTweetsCLF
import pandas as pd
import lightning as L
from dataloader import SpanishTweetsDataModule


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
# Make sure to provide the correct path to the trained model checkpoint
model_checkpoint = "spanish_tweets2/r18a2167/checkpoints/epoch=19-valid/average_final_metric=0.02.ckpt"
loaded_model = SpanishTweetsCLF.load_from_checkpoint(model_checkpoint)

trainer = L.Trainer(accelerator="cpu", devices=1)

train_dataset_path = "data/full_data/cleaned/train_clean_encoded.csv"
val_dataset_path = "data/full_data/cleaned/val_clean_encoded.csv"
test_dataset_path = "data/test_data/cleaned/cleaned_politicES_phase_2_test_public.csv"

dm = SpanishTweetsDataModule(train_dataset_path,
                             val_dataset_path,
                             test_dataset_path)

predictions = trainer.predict(loaded_model, dm)

with open("results.txt", "w") as f:
    f.write(str(predictions))
    print(predictions)