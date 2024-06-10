import torch
from utils.utils import calculate_metrics
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE


def eval(model, tokenizer, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []
    all_embeddings = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            print(inputs.shape)
            """ 
                We need to consider two separate cases: the one where the context is absent and the one where it is present.
                When we have the context, we need to concatenate them together and we use the special method encode_plus
            """
            if (len(inputs) > 1): # TODO: Check if this is correct
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
            outputs, last_hidden_state = model(**tokenized_inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels)
            all_predictions.extend(predicted)
            all_embeddings.append(last_hidden_state)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = all_embeddings.cpu().numpy()  # Convert to numpy array
    all_labels = torch.tensor(all_labels).cpu().numpy()

    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(all_predictions, all_labels)
    tsne_data = (all_embeddings, all_labels)
    return val_accuracy, val_precision, val_recall, val_f1, tsne_data


def predict_proba(input, model, tokenizer, device):
    """ Needed because SHAP passes a list with a single string as input and the tokenizer throws an error where there is a list with len == 1 as input """
    input = list(input) if isinstance(input, np.ndarray) else input # convert to list if it is numpy array else leave as it is
    model.eval()
    with torch.no_grad():
        tokenized_inputs = tokenizer(input, add_special_tokens=True, padding='longest', return_tensors='pt',
                                     max_length=512, truncation=True)
        tokenized_inputs = tokenized_inputs.to(device)
        outputs,_ = model(**tokenized_inputs)
        outputs = outputs.to('cpu')
        proba = outputs.softmax(dim=-1).detach().numpy()
    return proba




def eval_explanations(original_texts, no_rationales, only_rationales, model, tokenizer, device):
    comprehensiveness = []
    sufficiency = []
    index = 0
    for original_text, text_no_rationales, text_only_rationales in zip(original_texts, no_rationales, only_rationales):
        original_proba = predict_proba(original_text, model, tokenizer, device)
        no_rationales_proba = predict_proba(text_no_rationales, model, tokenizer, device) if text_no_rationales != "" else [[0,0,0]]
        only_rationales_proba = predict_proba(text_only_rationales, model, tokenizer, device) if text_only_rationales != "" else [[0,0,0]]
        print(" ONLY RATIONALES ",text_only_rationales)
        pred_id = np.argmax(original_proba)
        comprehensiveness.append(original_proba[0][pred_id] - no_rationales_proba[0][pred_id])
        sufficiency.append(original_proba[0][pred_id] - only_rationales_proba[0][pred_id])
        index += 1
    comprehensiveness_score = sum(comprehensiveness)/len(comprehensiveness)
    sufficiency_score = sum(sufficiency)/len(sufficiency)
    return comprehensiveness_score, sufficiency_score


