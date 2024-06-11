import eval.eval
from eval.eval import predict_proba
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shap
from explainability.lime_text import IndexedString


# Text.token_segments = custom_shap_token_segments # Monkey patch with SHAP because it doesn't support inputs that are multiple sequences separated by </s></s> tokens

def explain_lime(dataloader, explainer, top_labels, save_plot_folder, model, tokenizer, device):
    no_rationales = []
    only_rationales = []

    texts = [input[0][0] if len(input) < 2 else input[0][0] + '</s></s>' + input[1][0] for input, _ in tqdm(dataloader)]
    for idx, text in enumerate(texts):
        exp = explainer.explain_instance(text, predict_proba, model, tokenizer, device, top_labels=top_labels,
                                         num_features=60, num_samples=500)
        exp.save_to_file(f'{save_plot_folder}/{idx}.html')
        pred_id = np.argmax(exp.predict_proba)
        explanation_as_map = exp.as_map()[pred_id]
        """ We match the explanations with the tokens obtained by the tokenizer LIME library uses. Then we extract the rationales"""
        lime_tokenized_text = IndexedString(text, bow=False).inverse_vocab
        #print(lime_tokenized_text)
        #print(explanation_as_map)
        explanation_as_map = sorted(explanation_as_map, key=lambda x: x[0]) # the indexes of the map are not in the right order
        text_without_rationales = " ".join([t for t,(index,score) in zip(lime_tokenized_text, explanation_as_map) if score<=0])
        text_only_rationales = " ".join([t for t,(index,score) in zip(lime_tokenized_text, explanation_as_map) if score>0])
        no_rationales.append(text_without_rationales)
        only_rationales.append(text_only_rationales)

    return texts, no_rationales, only_rationales


def explain_shap(dataloader, explainer, model, save_plot_folder, tokenizer, device):
    no_rationales = []
    only_rationales = []
    # texts = [input[0][0] if len(input) < 2 else (input[0][0],input[1][0]) for input,_ in tqdm(dataloader)][:5] # just text or text+context
    texts = [input[0][0] if len(input) < 2 else input[0][0] + "</s></s>" + input[1][0] for input, _ in
             tqdm(dataloader)] # just text or text+context
    explanations = explainer(texts)
    shap_values = explanations.values

    for i, text in enumerate(texts):
        with open(f'{save_plot_folder}/{i}.html', 'w') as f:
            f.write(shap.text_plot(explanations[i], display=False))
        pred_classes = predict_proba(text, model, tokenizer, device)
        shap_score = [0] * shap_values[i].shape[0]
        """ Get the explanation for the majority class """
        pred_id = np.argmax(pred_classes)
        explanation = shap_values[i].T[
            pred_id]  # Shape shap_values[i]: len x num_classes, shap_values[i].T: num_classes x len
        for i, exp in enumerate(explanation):
            if (exp > 0):
                shap_score[i] = exp

        tokens = tokenizer(text, add_special_tokens=True, padding='longest', return_tensors='pt',
                           max_length=512, truncation=True)['input_ids'][0]
        text_without_rationales = [t1 for t1, t2 in zip(tokens, shap_score) if
                                   t2 == 0 or tokenizer.decode(t1) in ['<s>', '</s>']][1:-1]
        text_without_rationales = tokenizer.decode(text_without_rationales)
        text_only_rationales = [t1 for t1, t2 in zip(tokens, shap_score) if
                                t2 != 0 or tokenizer.decode(t1) in ['<s>', '</s>']][1:-1]
        text_only_rationales = tokenizer.decode(text_only_rationales)
        no_rationales.append(text_without_rationales)
        only_rationales.append(text_only_rationales)
    return texts, no_rationales, only_rationales
