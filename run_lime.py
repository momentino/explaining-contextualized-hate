from explainability.lime_text import LimeTextExplainer

import torch
from torch.utils.data import DataLoader

import argparse
import os
import yaml
import shutil

from model.model import RobertaForToxicClassification
from utils.utils import jsonl_to_df
from data.dataset import ToxicLangDataset
from eval.eval import eval_explanations
from explainability.explainability import explain_lime

from transformers import AutoTokenizer

import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset_file_path', type=str)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--context', action='store_true')
    # argument useful in case we want to use the whole dataset, rather than split automatically a whole dataset and just take the test portion
    parser.add_argument('--no_split', action='store_true')

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
    no_split = args.no_split
    results_file = config['explainability_results_path']
    dataset_file_path = args.dataset_file_path
    context = True if args.context else False

    device = torch.device(config['device'])

    """ Model """
    model = RobertaForToxicClassification(config['model'], config['n_class'])
    model = model.to(device)

    """ Tokenizer """
    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    """ Dataset preparation """
    dataset_path = dataset_file_path
    dataset_df = jsonl_to_df(dataset_path, s)

    if no_split:
        dataset = ToxicLangDataset(dataset_df=dataset_df, split='no_split', context=context)
    else:
        dataset = ToxicLangDataset(dataset_df=dataset_df, split='test', context=context)
    loader = DataLoader(dataset, batch_size=1)

    """ Get checkpoint path """
    checkpoint_path = args.checkpoint_path

    """ Evaluate """
    model.load_state_dict(torch.load(checkpoint_path))  # Load best model

    save_explanation_plots_folder = os.path.join(config['save_plot_folder_lime'],
                                                 "context" if context else "no_context")
    if not os.path.isdir(save_explanation_plots_folder):
        os.mkdir(save_explanation_plots_folder)
    explainer = LimeTextExplainer(class_names=config['class_names'], bow=False)
    original_texts, no_rationales, only_rationales = explain_lime(loader, explainer, config['n_class'],
                                                                  save_explanation_plots_folder,
                                                                  model,
                                                                  tokenizer,
                                                                  device)

    comprehensiveness, sufficiency = eval_explanations(original_texts, no_rationales, only_rationales, model, tokenizer,
                                                       device)
    print(" Quality of the explanations evaluated. Comprehensiveness: {}, Sufficiency: {}".format(comprehensiveness,
                                                                                                  sufficiency))
    df = pd.read_csv(results_file)
    results_row = [context, 'LIME', comprehensiveness, sufficiency]
    combined_data = pd.concat([df, pd.DataFrame([results_row],
                                                columns=["context", "exp_method", "comprehensiveness", "sufficiency"])],
                              ignore_index=True)
    combined_data.to_csv(results_file, index=False)

    """ Save the folder with the explanations to ZIP so that we can get it when running in the Colab """
    shutil.make_archive(f'{config["save_plot_folder_lime"]}/plots_{"context" if context else "no_context"}', 'zip',
                        save_explanation_plots_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explaining Contextualized Hate', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
