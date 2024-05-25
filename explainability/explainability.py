from eval.eval import predict_proba
from tqdm import tqdm
import numpy as np

def explain_lime(dataloader, explainer, top_labels, model, tokenizer, device, topk=2):

    results = {}
    res_list = []
    for input, label in tqdm(dataloader):
        if(len(input) > 1):
            text = input[0][0] + '[SEP]' + input[1][0]
        else:
            text = input[0][0]
        print(text)
        exp = explainer.explain_instance(text, predict_proba, model, tokenizer, device, top_labels=top_labels, num_features=500, num_samples=2000)

        pred_id = np.argmax(exp.predict_proba)
        print("PRED ID ", pred_id)
        print(" TRUE LABEL ",label)
        results["classification"] = pred_id

        lime_score = [0] * len(text.split(" "))

        explanation = exp.as_map()[pred_id]
        for exp in explanation:
            print("EXP ",exp)
            if (exp[1] > 0):
                lime_score[exp[0]] = exp[1]


        final_explanation = [0]
        tokens = text.split(" ")
        for i in range(len(tokens)):
            temp_tokens = tokenizer.encode(tokens[i], add_special_tokens=False)
            for j in range(len(temp_tokens)):
                final_explanation.append(lime_score[i])
        final_explanation.append(0)
        lime_score = final_explanation

        topk_indicies = sorted(range(len(lime_score)), key=lambda i: lime_score[i])[-topk:]

        hard_rationales = []
        for ind in topk_indicies:
            hard_rationales.append({'end_token': ind + 1, 'start_token': ind})

        results["rationales"] = [{"hard_rationale_predictions": hard_rationales,
                               "soft_rationale_predictions": lime_score}]
        res_list.append(results)

    return res_list
