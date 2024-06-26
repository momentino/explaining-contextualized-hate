import torch
from torch.utils.data import DataLoader

import argparse
import os
import yaml

from model.model import RobertaForToxicClassification
from utils.utils import get_optimizer, jsonl_to_df, get_loss_function, load_config
from data.dataset import ToxicLangDataset
from train.train import train

from transformers import AutoTokenizer


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset_file_path', type=str)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--context', action='store_true')
    return parser


# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config


def main(args):
    config_path = 'config'
    config = load_config(config_path,
                         'config.yaml')  # load the configuration file (the parameters will then be used like a dictionary with key-value pairs
    s = args.random_seed
    dataset_file_path = args.dataset_file_path
    context = True if args.context else False

    device = torch.device(config['device'])

    """ Model """
    model = RobertaForToxicClassification(config['model'], config['n_class'])
    model = model.to(device)
    model_save_path_base = config['model_save_path_base']
    if not os.path.exists(model_save_path_base):
        os.makedirs(model_save_path_base)

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

    train_dataset = ToxicLangDataset(dataset_df=dataset_df, split='train', context=context)
    val_dataset = ToxicLangDataset(dataset_df=dataset_df, split='val', context=context)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    model_save_path = os.path.join(model_save_path_base, f'yu22_{s}.pth')
    """ Train """
    train(model, tokenizer, train_loader, val_loader, config['training_epochs'], optimizer, criterion, model_save_path,
          device)
    print(" Fine-tuning completed. ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explaining Contextualized Hate', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
