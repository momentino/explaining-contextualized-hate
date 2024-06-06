import eval.eval
from eval.eval import predict_proba
from tqdm import tqdm
import numpy as np

def explain_lime(dataloader, explainer, top_labels, model, tokenizer, device):

    res_list = []
    texts = [input[0][0] if len(input) > 1 else input[0][0] + input[1][0] for input, _ in tqdm(dataloader)]
    for text in texts:
        exp = explainer.explain_instance(text, predict_proba, model, tokenizer, device, top_labels=top_labels, num_features=60, num_samples=500)

        pred_id = np.argmax(exp.predict_proba)
        #print(" TRUE LABEL ",label)
        #results["classification"] = pred_id
        #print(tokenizer(text, add_special_tokens=True, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'])
        lime_score = [0] * len(tokenizer(text, add_special_tokens=False, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0])
        #print(" LIME SCORE LEN ",len(lime_score))
        explanation = exp.as_map()[pred_id]
        #print(" EXP LEN ",len(explanation), explanation)
        #print(" TEXT LEN ",len(tokenizer(text, add_special_tokens=True, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0])," ",)
        #print(" EXPLAINABILITY ", tokenizer(text, add_special_tokens=True, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0])
        for exp in explanation:
            if (exp[1] > 0):
                lime_score[exp[0]] = exp[1]
        #print("LIME SCORE ",lime_score)

        """final_explanation = [0]
        tokens = tokenizer(text, add_special_tokens=False, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0]
        for i in range(len(tokens)):
            #temp_tokens = tokenizer.encode(tokens[i], add_special_tokens=False)
            #for j in range(len(temp_tokens)):
            final_explanation.append(lime_score[i])
        final_explanation.append(0)
        lime_score = final_explanation"""
        res_list.append(lime_score)

    return res_list

def explain_shap(dataloader, explainer, model, tokenizer, device):
    res_list = []
    texts = [input[0][0] if len(input) > 1 else input[0][0] + input[1][0] for input,_ in tqdm(dataloader)] # just text or text+context
    texts = texts[:2]
    exp = explainer(texts)
    shap_values = exp.values
    for i, text in enumerate(texts):
        pred_classes = predict_proba(text, model, tokenizer, device)
        shap_score = [0] * shap_values[i].shape[0]
        """ Get the explanation for the majority class """
        pred_id = np.argmax(pred_classes)
        explanation = shap_values[i].T[pred_id] # Shape shap_values[i]: len x num_classes, shap_values[i].T: num_classes x len
        print(explanation)
        for exp in explanation:
            if (exp[1] > 0):
                shap_score[exp[0]] = exp[1]
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