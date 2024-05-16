import torch
from torch.utils.data import Dataset

import json
import pandas as pd

from sklearn.utils import shuffle


class ToxicLangDataset(Dataset):
    def __init__(self, dataset_df, split, random_seed, context, dataset_name):
        self.split = split
        self.random_seed = random_seed  # seed that should be selected randomly. We should perform 10 different tests with 10 seeds and average.
        self.df = dataset_df[:100]

        self.context = context

        train_limit = int(0.8 * len(self.df))
        val_limit = int(0.9 * len(self.df))
        # used to split the dataset
        split_separators = {
            'train': (1,train_limit),
            'val': (train_limit, val_limit),
            'test': (val_limit, len(self.df))
        }
        if(context):
            if(dataset_name=="pav20"):
                self.df = self.df[self.df['context'] != ''] #take only those where context is not
            self.contexts = self.df['context'][split_separators[split][0]:split_separators[split][1]].to_list()
        else:
            if (dataset_name == "pav20"):
                self.df = self.df[self.df['context'] == '']  # take only those where context is not
        self.texts = self.df['target'][split_separators[split][0]:split_separators[split][1]].to_list()
        self.labels = self.df['label'][split_separators[split][0]:split_separators[split][1]].to_list()



    def _jsonl_to_df(self, file, random_seed):
        data = []
        with open(file) as f:
            for line in f:
                data.append(json.loads(line))

        df = pd.json_normalize(data)
        df = shuffle(df, random_state=random_seed)
        return df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if(self.context):
            text = [self.texts[idx],self.contexts[idx]]
        else:
            text = [self.texts[idx]]
        label = self.labels[idx]
        label = torch.tensor(int(label))
        return text, label