import torch
from utils.utils import calculate_metrics

from model.model import RobertaForToxicClassification
from transformers import AutoTokenizer

from tqdm import tqdm
import os
from utils.utils import load_config


def eval(model, tokenizer, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            """ 
                We need to consider two separate cases: the one where the context is absent and the one where it is present.
                When we have the context, we need to concatenate them together and we use the special method encode_plus
            """
            if (len(inputs) > 1):
                tokenized_inputs = tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(c,t) for c,t in zip(inputs[0], inputs[1])],  # target
                    add_special_tokens=True,  # Add [CLS] and [SEP]
                    return_tensors='pt',  # Return PyTorch tensors
                    padding='longest',  # Pad to the maximum length
                    max_length=512,
                    truncation=True
                )
            else:
                tokenized_inputs = tokenizer(inputs[0], padding='longest', return_tensors='pt', max_length=512, truncation=True)
            tokenized_inputs = tokenized_inputs.to(device)
            labels = labels.to(device)
            outputs = model(**tokenized_inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels)
            all_predictions.extend(predicted)

    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(all_predictions, all_labels)

    return val_accuracy, val_precision, val_recall, val_f1

def predict_proba(input, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        tokenized_inputs = tokenizer(input, add_special_tokens=True, padding='longest', return_tensors='pt', max_length=512, truncation=True)
        tokenized_inputs = tokenized_inputs.to(device)
        outputs = model(**tokenized_inputs)
        proba = outputs.logits.softmax(dim=-1).detach().numpy()


    return proba