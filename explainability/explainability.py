from eval.eval import predict_proba
from tqdm import tqdm

def explain_lime(dataloader, explainer, model, tokenizer, device):
    explanations = []
    for input, _ in tqdm(dataloader):
        if(len(input[0]) > 1):
            text = input[0][0] + '[SEP]' + input[0][1]
        else:
            text = input[0][0]
        single_sample_exp = explainer.explain_instance(text, predict_proba, model, tokenizer, device, num_features=20, num_samples=2000)
        explanations.append(single_sample_exp)
    return explanations
