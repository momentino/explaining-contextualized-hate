from explainability.lime_text import LimeTextExplainer

import torch
from torch.utils.data import DataLoader

import argparse
import os
import yaml

from model.model import RobertaForToxicClassification
from utils.utils import jsonl_to_df
from data.dataset import ToxicLangDataset
from eval.eval import eval_explanations
from explainability.explainability import explain_lime

from transformers import AutoTokenizer

import csv
import pandas as pd

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset_file_path', type=str)
    parser.add_argument('--saved_model_name', type=str)
    parser.add_argument('--context', action='store_true')

    return parser

# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def main(args):
    config_path = 'config'
    config = load_config(config_path, 'config.yaml') # load the configuration file (the parameters will then be used like a dictionary with key-value pairs

    results_file = config['explainability_results_path']
    dataset_file_path = args.dataset_file_path
    context = True if args.context else False

    dataset_name = "yu22" if "yu" in dataset_file_path else "pavlopoulos20"

    device = torch.device(config['device'])

    """ Model """
    model = RobertaForToxicClassification(config['model'],config['n_class_'+dataset_name])
    model = model.to(device)
    model_save_path = config['model_save_path']
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    """ Tokenizer """
    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    """ Dataset preparation """
    dataset_path = dataset_file_path
    dataset_df = jsonl_to_df(dataset_path)


    dataset = ToxicLangDataset(dataset_df=dataset_df, split='no_split', context=context, dataset_name=dataset_name)

    loader = DataLoader(dataset, batch_size=1)

    """ Define name of model """
    model_name = args.saved_model_name


    """ Evaluate """
    model.load_state_dict(torch.load(os.path.join(model_save_path, model_name))) # Load best model


    explainer = LimeTextExplainer(class_names=config['class_names_'+dataset_name])
    explanations = explain_lime(loader, explainer, config['n_class_'+dataset_name], model,  tokenizer, device)

    comprehensiveness, sufficiency = eval_explanations(loader, explanations, model, tokenizer, device)
    print(" Quality of the explanations evaluated. Comprehensiveness: {}, Sufficiency: {}".format(comprehensiveness,sufficiency))
    df = pd.read_csv(results_file)

    results_row = [dataset_name, context, 'LIME',comprehensiveness,sufficiency]

    combined_data = pd.concat([df, pd.DataFrame([results_row],
                                                columns=["dataset","context","exp_method","comprehensiveness","sufficiency"])], ignore_index=True)
    combined_data.to_csv(results_file, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explaining Contextualized Hate', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)