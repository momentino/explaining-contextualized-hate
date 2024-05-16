import torch
from torch.utils.data import DataLoader

import argparse
import os
import yaml

from model.model import RobertaForToxicClassification
from utils.utils import get_optimizer, jsonl_to_df, get_loss_function
from data.dataset import ToxicLangDataset
from train.train import train
from eval.eval import eval

from transformers import AutoTokenizer

import csv

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset_file_path', type=str)
    parser.add_argument('--random_seed', type=str)
    parser.add_argument('--context', type=bool)

    return parser

# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def main(args):
    config_path = 'config'
    config = load_config(config_path, 'config.yaml') # load the configuration file (the parameters will then be used like a dictionary with key-value pairs
    s = args.random_seed
    results_file = config['results_path']
    dataset_file_path = args.dataset_file_path
    context = args.context

    dataset_name = "yu22" if "yu" in dataset_file_path else "pavlopoulos20"

    device = torch.device(config['device'])

    """ Model """
    model = RobertaForToxicClassification(config['model'],config['n_class_'+dataset_name])

    """ Optimizer """
    optimizer_name = config['optimizer']
    model_parameters = model.parameters()
    lr = config['lr']
    weight_decay = config['weight_decay']
    optimizer = get_optimizer(optimizer_name, model_parameters, lr=lr, weight_decay=weight_decay)

    """ Loss function"""
    loss_name = config['criterion']
    criterion = get_loss_function(loss_name)

    """ Tokenizer """
    tokenizer = AutoTokenizer.from_pretrained(config['model'])


    """ Dataset preparation """
    dataset_path = dataset_file_path
    dataset_df = jsonl_to_df(dataset_path, s)

    train_dataset = ToxicLangDataset(dataset_df=dataset_df, split='train', random_seed=s, context=context, dataset_name=dataset_name)
    val_dataset = ToxicLangDataset(dataset_df=dataset_df, split='val', random_seed=s, context=context, dataset_name=dataset_name)
    test_dataset = ToxicLangDataset(dataset_df=dataset_df, split='test', random_seed=s, context=context, dataset_name=dataset_name)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    """ Define name of model """
    model_name = "yu22_" + s if "yu" in dataset_path else "pav20_" + s
    """ Train """
    train(model, tokenizer, train_loader, val_loader, config['training_epochs'],optimizer, criterion, config['model_save_path'])

    """ Evaluate """
    model.load_state_dict(torch.load(os.path.join(config['model_save_path'], model_name))) # Load best model
    test_accuracy, test_precision, test_recall, test_f1 = eval(model, tokenizer, test_loader)

    """ Save results """
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([dataset_name, model_name, s, test_accuracy, test_precision, test_recall, test_f1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explaining Contextualized Hate', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)