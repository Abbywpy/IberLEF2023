from trainer import SpanishTweetsCLF
import argparse
import torch
import pandas as pd


def create_probs_dict_tweet_level(logits, attr):
    probs_dict_tweet_level = {}
    for i, a in enumerate(attr):
        lgts = logits[i]
        probs = torch.softmax(lgts, dim=1)
        probs_dict_tweet_level[a] = probs
    return probs_dict_tweet_level

def create_probs_dict_cluster_level(probs_dict_tweet_level, cluster_size=80):
    probs_dict_cluster_level = {}
    for attr in probs_dict_tweet_level:
        probs = probs_dict_tweet_level[attr]
        probs_cluster_level = torch.mean(torch.reshape(probs, (int(probs.shape[0]/cluster_size), cluster_size, -1)), dim=1)
        probs_dict_cluster_level[attr] = probs_cluster_level
    return probs_dict_cluster_level

def create_class_preds_dict_cluster_level(probs_dict_cluster_level):
    class_preds_dict_cluster_level = {}
    for attr in probs_dict_cluster_level:
        probs = probs_dict_cluster_level[attr]
        predictions_cluster_level = torch.argmax(probs, dim=1)
        class_preds_dict_cluster_level[attr] = predictions_cluster_level
    return class_preds_dict_cluster_level

def create_decoding_dict():
    decoding_dict = {}
    decoding_dict['gender'] = {0: 'female', 1: 'male'}
    decoding_dict['profession'] = {0: 'celebrity', 1: 'journalist', 2: 'politician'}
    decoding_dict['ideology_binary'] = {0: 'left', 1: 'right'}
    decoding_dict['ideology_multiclass'] = {0: 'left', 1: 'moderate_left', 2: 'moderate_right', 3: 'right'}
    return decoding_dict

def decode_preds(preds, decoding_dict):
    decoded_preds = {}
    for attr in preds:
        decoded_preds[attr] = [decoding_dict[attr][p] for p in preds[attr].numpy()]
    return decoded_preds


def create_submission_file(decoded_preds, output_path, list_of_labels = None):
    df = pd.DataFrame()
    if list_of_labels:
        df['label'] = list_of_labels
    else:
        df['label'] = [i for i in range(len(next(iter(decoded_preds.values()))))]
    for attr in decoded_preds:
        df[attr] = decoded_preds[attr]
    print(df)
    df.to_csv(output_path, index=False)



def main(args):
    attr = ['gender', 'profession', 'ideology_binary', 'ideology_multiclass']
    input_df = pd.read_csv(args.input)
    model = SpanishTweetsCLF.load_from_checkpoint(args.chkpt)

    model.eval()
    with torch.no_grad():
        logits = model(input_df['cleaned_tweet'])
    
    probs_dict_tweet_level = create_probs_dict_tweet_level(logits, attr)

    probs_dict_cluster_level = create_probs_dict_cluster_level(probs_dict_tweet_level, cluster_size=80)

    class_preds_dict_cluster_level = create_class_preds_dict_cluster_level(probs_dict_cluster_level)

    decoding_dict = create_decoding_dict()

    decoded_preds = decode_preds(class_preds_dict_cluster_level, decoding_dict)

    labels = input_df['label'].unique().tolist()
    create_submission_file(decoded_preds, args.output, list_of_labels = labels)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/test_data/cleaned/cleaned_politicES_phase_2_test_public.csv", help="path to test data")
    parser.add_argument("--output", type=str, default="data/test_data/output/results.csv", help="path to model output")
    parser.add_argument("--chkpt", type=str, default="checkpoints/best_model.ckpt", help="path to model checkpoint")
    args = parser.parse_args()

    main(args)