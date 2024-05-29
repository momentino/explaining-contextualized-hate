from eval.eval import predict_proba
from tqdm import tqdm
import numpy as np

def explain_lime(dataloader, explainer, top_labels, model, tokenizer, device):

    results = {}
    res_list = []
    for input, label in tqdm(dataloader):
        print(label)
        if(label[0] == 1):
            print(" Avoid sample ")
            continue
        if(len(input) > 1):
            print(input)
            text = input[0][0] + input[1][0]
        else:
            text = input[0][0]
        exp = explainer.explain_instance(text, predict_proba, model, tokenizer, device, top_labels=top_labels, num_features=50, num_samples=500)

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

        final_explanation = [0]
        tokens = tokenizer(text, add_special_tokens=False, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0]
        for i in range(len(tokens)):
            #temp_tokens = tokenizer.encode(tokens[i], add_special_tokens=False)
            #for j in range(len(temp_tokens)):
            final_explanation.append(lime_score[i])
        final_explanation.append(0)
        lime_score = final_explanation



        #topk_indicies = sorted(range(len(lime_score)), key=lambda i: lime_score[i])[-topk:]

        """hard_rationales = []
        for ind in topk_indicies:
            hard_rationales.append({'end_token': ind + 1, 'start_token': ind})
        print(" HARD RATIONALES ",hard_rationales)"""
        #results["lime_score"] = lime_score
        res_list.append(lime_score)

    return res_list
