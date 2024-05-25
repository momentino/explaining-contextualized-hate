import torch
from utils.utils import calculate_metrics

from model.model import RobertaForToxicClassification
from transformers import AutoTokenizer

from tqdm import tqdm
import os
from utils.utils import load_config
import numpy as np

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
        print(" IN PREDICT PROBA ", input)
        print(" IN PREDICT PROBA ", tokenizer.decode(tokenized_inputs['input_ids'][0]))
        tokenized_inputs = tokenized_inputs.to(device)
        outputs = model(**tokenized_inputs)
        outputs = outputs.to('cpu')
        proba = outputs.softmax(dim=-1).detach().numpy()


    return proba

def eval_explanations(dataloader, rationales, model, tokenizer, device):
    comprehensiveness = []
    sufficiency = []
    index = 0
    for input, label in tqdm(dataloader):
        if(len(input) > 1):
            original_text = input[0][0] + '</s><s>' + input[1][0]
        else:
            original_text = input[0][0]
        tokens = tokenizer(original_text, add_special_tokens=False, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0]
        #print(" TOKENS ", len(tokens))
        #print(" RATIONALES ",len(rationales[index]))
        text_without_rationales = [t1 for t1, t2 in zip(tokens, rationales[index]) if t2 == 0 or tokenizer.decode(t1) in ['<s>','</s>']]
        text_without_rationales = tokenizer.decode(text_without_rationales)
        only_rationales = [t1 for t1, t2 in zip(tokens, rationales[index]) if t2 != 0 or tokenizer.decode(t1) in ['<s>','</s>']]
        only_rationales = tokenizer.decode(only_rationales)
        #print(" ORIGINAL ", original_text)
        #print(" NO RATIONALES ", text_without_rationales)
        #print(" ONLY RATIONALES ",only_rationales)

        original_proba = predict_proba(original_text, model, tokenizer, device)
        no_rationales_proba = predict_proba(text_without_rationales, model, tokenizer, device)
        only_rationales = predict_proba(only_rationales, model, tokenizer, device)

        pred_id = np.argmax(original_proba)
        #print(" ORIGINAL PROBA ",original_proba[0][pred_id])
        #print(" NO RATIONALES PROBA ", no_rationales_proba[0][pred_id])
        comprehensiveness.append(original_proba[0][pred_id] - no_rationales_proba[0][pred_id])
        sufficiency.append(original_proba[0][pred_id] - only_rationales[0][pred_id])

        index+=1
    comprehensiveness_score = sum(comprehensiveness)/len(comprehensiveness)
    sufficiency_score = sum(sufficiency)/len(sufficiency)
    return comprehensiveness_score, sufficiency_score


