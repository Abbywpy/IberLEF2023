import torch
from torch import tensor
from collections import Counter
from dataloader import SpanishTweetsDataModule
from trainer import SpanishTweetsCLF
import pandas as pd
import lightning as L
import argparse


def create_argumentparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", "-cp", type=str, required=True, default="spanish_tweets2/r18a2167/checkpoints/epoch=19-valid/average_final_metric=0.02.ckpt")
    parser.add_argument("--test_dataset_path", "-tdp", type=str, required=True, default="data/test_data/cleaned/cleaned_politicES_phase_2_test_public.csv")
    parser.add_argument("--output_path", "-op", type=str, required=True, default="results.csv")

    return parser


def create_decoding_dict():
    decoding_dict = {}
    decoding_dict['gender'] = {0: 'female', 1: 'male'}
    decoding_dict['profession'] = {0: 'celebrity', 1: 'journalist', 2: 'politician'}
    decoding_dict['ideology_binary'] = {0: 'left', 1: 'right'}
    decoding_dict['ideology_multiclass'] = {0: 'left', 1: 'moderate_left', 2: 'moderate_right', 3: 'right'}

    return decoding_dict


def decode_preds(predictions, decoding_dict):
    attrs = {'gender': [], 'profession': [], 'ideology_binary': [], 'ideology_multiclass': []}
    for i in range(len(predictions)):  # for each batch = cluster
        for j, attr in enumerate(list(attrs.keys())):  # for each attribute
            pred_idcs = []
            for k in range(len(predictions[i][j])):  # for each tweet in batch
                pred_idx = torch.argmax(predictions[i][j][k])
                pred_idcs.append(pred_idx)
            pred_counter = Counter(pred_idcs)  # count individual predictions for each tweet and trait
            most_common_pred = pred_counter.most_common(1)[0][0]  # get most common prediction
            attrs[f"{attr}"] += [decoding_dict[f"{attr}"][int(most_common_pred)] for _ in range(
                len(pred_counter))]  # append to dictionary as many times as there are tweets in batch

    attrs = handle_ideology_mismatch(attrs)

    return attrs


def handle_ideology_mismatch(attrs):
    attrs['ideology_multiclass'] = ['moderate_right' if 'right' in attrs['ideology_binary'] and "left" in attrs['ideology_multiclass'] else _ for _ in attrs['ideology_multiclass']]
    attrs['ideology_multiclass'] = ['moderate_left' if 'left' in attrs['ideology_binary'] and "right" in attrs['ideology_multiclass'] else _ for _ in attrs['ideology_multiclass']]
    return attrs


def main():
    parser = create_argumentparser()
    args = parser.parse_args()

    loaded_model = SpanishTweetsCLF.load_from_checkpoint(args.model_checkpoint)

    trainer = L.Trainer(accelerator="gpu", devices=1)

    train_dataset_path = "data/full_data/cleaned/train_clean_encoded.csv"
    val_dataset_path = "data/full_data/cleaned/val_clean_encoded.csv"
    test_dataset_path = args.test_dataset_path

    dm = SpanishTweetsDataModule(train_dataset_path,
                                 val_dataset_path,
                                 test_dataset_path)

    predictions = trainer.predict(loaded_model, dm)

    attrs = {'gender': [], 'profession': [], 'ideology_binary': [], 'ideology_multiclass': []}

    decoding_dict = create_decoding_dict()

    attrs.update(decode_preds(predictions, decoding_dict))

    results_df = pd.DataFrame(attrs)

    test_df = pd.read_csv(test_dataset_path)
    labels = test_df[["label"]]

    combined_df = pd.concat([labels, results_df], axis=1)

    combined_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
