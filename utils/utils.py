import torch
import torch.optim as optim
import torch.nn as nn

import json
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import yaml

from shap.utils.transformers import (
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
    parse_prefix_suffix_for_tokenizer,
)
from shap.utils import safe_isinstance


def get_optimizer(optimizer_name: str, parameters, **kwargs):
    optimizer_class = getattr(optim, optimizer_name)
    return optimizer_class(parameters, **kwargs)

def get_loss_function(loss_name: str):
    loss_class = getattr(nn, loss_name)
    return loss_class()

def jsonl_to_df(file, random_seed=-1):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.json_normalize(data)
    if(random_seed is not -1):
        df = shuffle(df, random_state=int(random_seed))
    return df

def calculate_metrics(predictions, targets):
    predictions = torch.tensor(predictions).cpu().numpy()
    targets = torch.tensor(targets).cpu().numpy()
    accuracy = (predictions == targets).mean()
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')

    return accuracy, precision, recall, f1

""" Converts a dataset split back to JSONL """
def dataset_to_jsonl(df, split, path,context, dataset_name):
    if (context):
        if (dataset_name == "pavlopoulos20"):
            df = df[df['context'].notna()]  # take only those where context is not
    else:
        if (dataset_name == "pavlopoulos20"):
            df = df[df['context'].isna()]  # take only those where context is not
    train_limit = int(0.8 * len(df))
    val_limit = int(0.9 * len(df))
    split_separators = {
        'train': (1, train_limit),
        'val': (train_limit, val_limit),
        'test': (val_limit, len(df))
    }
    selected_rows = df.iloc[split_separators[split][0]:split_separators[split][1]]
    with open(path, 'w') as jsonl_file:
        for index, row in selected_rows.iterrows():
            row_dict = row.to_dict()
            jsonl_file.write(json.dumps(row_dict) + '\n')

# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config


def custom_shap_token_segments(self, s):
    """ Returns the substrings associated with each token in the given string.
    """

    try:
        if(isinstance(s, tuple) and len(s) == 2):
            token_data = self.tokenizer(s[0],s[1], return_offsets_mapping=True)
            offsets = token_data["offset_mapping"]
            offsets = [(0, 0) if o is None else o for o in offsets]
            # get the position of the first special character </s> because we need to find the end of the first sentence and the beginning of the second
            offset_pos_input_sep = [i for i, n in enumerate(offsets) if n == (0,0)][1]
            parts = ([s[0][offsets[i][0]:max(offsets[i][1], offsets[i + 1][0])] for i in range(offset_pos_input_sep)] +
                     [s[1][offsets[i][0]:max(offsets[i][1], offsets[i + 1][0])] for i in range(offset_pos_input_sep,len(offsets) - 1)])
            parts.append(s[1][offsets[len(offsets) - 1][0]:offsets[len(offsets) - 1][1]])
        elif (isinstance(s,str)):
            token_data = self.tokenizer(s, return_offsets_mapping=True)
            offsets = token_data["offset_mapping"]
            offsets = [(0, 0) if o is None else o for o in offsets]

            parts = [s[offsets[i][0]:max(offsets[i][1], offsets[i+1][0])] for i in range(len(offsets)-1)]
            parts.append(s[offsets[len(offsets)-1][0]:offsets[len(offsets)-1][1]])
        return parts, token_data["input_ids"]
    except (NotImplementedError, TypeError): # catch lack of support for return_offsets_mapping
        if (isinstance(s, tuple) and len(s) == 2):
            token_ids = self.tokenizer(s[0],s[1])['input_ids']
        elif (isinstance(s, str)):
            token_ids = self.tokenizer(s)['input_ids']
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        else:
            tokens = [self.tokenizer.decode([id]) for id in token_ids]
        if hasattr(self.tokenizer, "get_special_tokens_mask"):
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(token_ids, already_has_special_tokens=True)
            # avoid masking separator tokens, but still mask beginning of sentence and end of sentence tokens
            special_keep = [getattr_silent(self.tokenizer, 'sep_token'), getattr_silent(self.tokenizer, 'mask_token')]
            for i, v in enumerate(special_tokens_mask):
                if v == 1 and (tokens[i] not in special_keep or i + 1 == len(special_tokens_mask)):
                    tokens[i] = ""

        # add spaces to separate the tokens (since we want segments not tokens)
        if safe_isinstance(self.tokenizer, SENTENCEPIECE_TOKENIZERS):
            for i, v in enumerate(tokens):
                if v.startswith("_"):
                    tokens[i] = " " + tokens[i][1:]
        else:
            for i, v in enumerate(tokens):
                if v.startswith("##"):
                    tokens[i] = tokens[i][2:]
                elif v != "" and i != 0:
                    tokens[i] = " " + tokens[i]

        return tokens, token_ids
