import torch
import torch.optim as optim
import torch.nn as nn

import json
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score


def get_optimizer(optimizer_name: str, parameters, **kwargs):
    optimizer_class = getattr(optim, optimizer_name)
    return optimizer_class(parameters, **kwargs)

def get_loss_function(loss_name: str):
    loss_class = getattr(nn, loss_name)
    return loss_class()

def jsonl_to_df(file, random_seed):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.json_normalize(data)
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
def dataset_to_jsonl(df, split, path):
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