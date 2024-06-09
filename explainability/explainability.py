import eval.eval
from eval.eval import predict_proba
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils.utils import custom_shap_token_segments

from shap.maskers._text import Text
#Text.token_segments = custom_shap_token_segments # Monkey patch with SHAP because it doesn't support inputs that are multiple sequences separated by </s></s> tokens

def explain_lime(dataloader, explainer, top_labels, save_plot_folder, model, tokenizer, device):

    res_list = []
    texts = [input[0][0] if len(input) < 2 else input[0][0] + '</s></s>' + input[1][0] for input, _ in tqdm(dataloader)]
    for idx,text in enumerate(texts[:2]):
        exp = explainer.explain_instance(text[:50], predict_proba, model, tokenizer, device, top_labels=top_labels, num_features=60, num_samples=500)
        exp.save_to_file(f'{save_plot_folder}/{idx}.html')
        pred_id = np.argmax(exp.predict_proba)
        #print(tokenizer.tokenize(text))

        lime_score = [0] * len(tokenizer(text, add_special_tokens=False, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0])
        explanation = exp.as_list(label=pred_id)
        explanation_as_map = exp.as_map()[pred_id]
        #for exp in explanation:
        #    if (exp[1] > 0):
        #        lime_score[exp[0]] = exp[1]

        """final_explanation = [0]
        tokens = tokenizer(text, add_special_tokens=False, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0]
        for i in range(len(tokens)):
            #temp_tokens = tokenizer.encode(tokens[i], add_special_tokens=False)
            #for j in range(len(temp_tokens)):
            final_explanation.append(lime_score[i])
        final_explanation.append(0)
        lime_score = final_explanation"""
        res_list.append(explanation)

    return res_list

def explain_shap(dataloader, explainer, model, save_plot_folder,tokenizer, device):
    res_list = []
    #texts = [input[0][0] if len(input) < 2 else (input[0][0],input[1][0]) for input,_ in tqdm(dataloader)][:5] # just text or text+context
    texts = [input[0][0] if len(input) < 2 else input[0][0] + "</s></s>" + input[1][0] for input, _ in tqdm(dataloader)]  # just text or text+context
    explanations = explainer(texts)
    shap_values = explanations.values


    for i, text in enumerate(texts):
        with open(f'{save_plot_folder}/{i}.html', 'w') as f:
            f.write(shap.text_plot(explanations[i], display=False))
        pred_classes = predict_proba(text, model, tokenizer, device)
        shap_score = [0] * shap_values[i].shape[0]
        """ Get the explanation for the majority class """
        pred_id = np.argmax(pred_classes)
        explanation = shap_values[i].T[pred_id] # Shape shap_values[i]: len x num_classes, shap_values[i].T: num_classes x len
        for i,exp in enumerate(explanation):
            if (exp > 0):
                shap_score[i] = exp
        """final_explanation = [0]
        tokens = tokenizer(text, add_special_tokens=False, padding='longest', return_tensors='pt', max_length=512,
                           truncation=True)['input_ids'][0]
        for i in range(len(tokens)):
            # temp_tokens = tokenizer.encode(tokens[i], add_special_tokens=False)
            # for j in range(len(temp_tokens)):
            final_explanation.append(shap_score[i])
        final_explanation.append(0)
        shap_score = final_explanation"""
        res_list.append(shap_score)
    return  res_list