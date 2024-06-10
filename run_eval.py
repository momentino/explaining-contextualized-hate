import torch
from torch.utils.data import DataLoader

import argparse
import os
import yaml
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt



from model.model import RobertaForToxicClassification
from utils.utils import jsonl_to_df, dataset_to_jsonl, load_config
from data.dataset import ToxicLangDataset
from eval.eval import eval

from transformers import AutoTokenizer



def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset_file_path', type=str)
    parser.add_argument('--random_seed', type=str)
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--ignore_results', action='store_true',)
    return parser

# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def main(args):
    eval_mode = args.eval
    ignore_results = args.ignore_results
    config_path = 'config'
    config = load_config(config_path, 'config.yaml') # load the configuration file (the parameters will then be used like a dictionary with key-value pairs
    s = args.random_seed
    results_file = config['results_path']
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
    dataset_df = jsonl_to_df(dataset_path, s)

    test_dataset = ToxicLangDataset(dataset_df=dataset_df, split='test', context=context, dataset_name=dataset_name)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    """ Get checkpoint path """
    checkpoint_path = args.checkpoint_path

    """ Evaluate """
    model.load_state_dict(torch.load(checkpoint_path)) # Load best model
    test_accuracy, test_precision, test_recall, test_f1, (embeddings, labels) = eval(model, tokenizer, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}, '
          f'Test Precision: {test_precision:.2f}, '
          f'Test Recall: {test_recall:.2f}, '
          f'Test F1: {test_f1:.2f}, ')

    """ t-SNE """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the result
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')  # Replace `your_labels` with the actual labels
    plt.colorbar()
    plt.title("t-SNE visualization of RoBERTa embeddings")

    # Save folder for t-SNE plot
    save_plot_folder = f'results/tsne_plots_{"context" if context else "no_context"}'
    if not os.path.isdir(save_plot_folder):
        os.mkdir(save_plot_folder)
    plt.savefig(f'{save_plot_folder}/tsne.png')

    if not ignore_results:
        """ 
            Read the results, and if the current ones are the best, save the train,test,split for the explanation part
            We are going to use the best model in that experiment so we need to know its split because it is generated randomly every time. 
        """
        df = pd.read_csv(results_file)
        filtered_df = df[df['context'] == context]
        """ We need to see if it is the best among the one in the same category. We don't want to compare contextual with non contextual models' results. """
        if not (filtered_df['f1'] > test_f1).any():
            context_path = 'context' if context else 'no_context'
            dataset_to_jsonl(dataset_df, 'train', os.path.join('datasets/'+dataset_name+'/data/best_run_splits/'+context_path,'train.jsonl'),context, dataset_name)
            dataset_to_jsonl(dataset_df, 'val', os.path.join('datasets/'+dataset_name+'/data/best_run_splits/'+context_path,'val.jsonl'),context, dataset_name)
            dataset_to_jsonl(dataset_df, 'test', os.path.join('datasets/'+dataset_name+'/data/best_run_splits/'+context_path,'test.jsonl'),context, dataset_name)
        """ Save results """
        results_row = [dataset_name, context, s, test_accuracy, test_precision, test_recall, test_f1]

        combined_data = pd.concat([df, pd.DataFrame([results_row], columns=["dataset","context","model_name","seed","accuracy","precision","recall","f1"])], ignore_index=True)
        combined_data.to_csv(results_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explaining Contextualized Hate', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)