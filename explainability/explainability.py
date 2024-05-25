from eval.eval import predict_proba
from tqdm import tqdm
import numpy as np

def explain_lime(dataloader, explainer, top_labels, model, tokenizer, device):

    results = {}
    res_list = []
    for input, label in tqdm(dataloader):
        if(len(input) > 1):
            text = input[0][0] + '[SEP]' + input[1][0]
        else:
            text = input[0][0]
        exp = explainer.explain_instance(text, predict_proba, model, tokenizer, device, top_labels=top_labels, num_features=6, num_samples=2000)

        pred_id = np.argmax(exp.predict_proba)
        #print("PRED ID ", pred_id)
        #print(" TRUE LABEL ",label)
        results["classification"] = pred_id
        #print(tokenizer(text, add_special_tokens=True, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'])
        lime_score = [0] * len(tokenizer(text, add_special_tokens=True, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0])
        #print(" LIME SCORE LEN ",len(lime_score))
        explanation = exp.as_map()[pred_id]
        #print(" EXP LEN ",len(explanation))

        for exp in explanation:
            if (exp[1] > 0):
                #print(exp)
                lime_score[exp[0]] = exp[1]


        final_explanation = [0]
        tokens = tokenizer(text, add_special_tokens=True, padding='longest', return_tensors='pt', max_length=512, truncation=True)['input_ids'][0]
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
        results["lime_score"] = lime_score
        res_list.append(results)

    return res_list
