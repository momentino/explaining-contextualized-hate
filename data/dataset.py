import torch
from torch.utils.data import Dataset

import json
import pandas as pd

from sklearn.utils import shuffle


class ToxicLangDataset(Dataset):
    def __init__(self, dataset_df, split, context, dataset_name):
        self.split = split
        self.df = dataset_df

        self.context = context

        train_limit = int(0.8 * len(self.df))
        val_limit = int(0.9 * len(self.df))
        # used to split the dataset
        split_separators = {
            'train': (1,train_limit),
            'val': (train_limit, val_limit),
            'test': (val_limit, len(self.df)),
            'no_split': (1, len(self.df))
        }
        if(context):
            if(dataset_name=="pavlopoulos20"):
                self.df = self.df[self.df['context'].notna()] #take only those where context is not NaN
            self.contexts = self.df['context'][split_separators[split][0]:split_separators[split][1]].to_list()
        else:
            if (dataset_name == "pavlopoulos20"):
                self.df = self.df[self.df['context'].isna()]  # take only those where context is NaN
        self.texts = self.df['target'][split_separators[split][0]:split_separators[split][1]].to_list()
        self.labels = self.df['label'][split_separators[split][0]:split_separators[split][1]].to_list()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if(self.context):
            text = [self.contexts[idx],self.texts[idx]]
        else:
            text = [self.texts[idx]]
        label = self.labels[idx]
        label = torch.tensor(int(label))
        return text, label