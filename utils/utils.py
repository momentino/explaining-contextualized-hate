import torch
import torch.optim as optim
import torch.nn as nn

import json
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import yaml


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


# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config
