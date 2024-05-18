import torch
from utils.utils import calculate_metrics

from tqdm import tqdm

def eval(model, tokenizer, val_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            """ 
                We need to consider two separate cases: the one where the context is absent and the one where it is present.
                When we have the context, we need to concatenate them together and we use the special method encode_plus
            """
            if (len(inputs) > 0):
                tokenized_inputs = tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(t, c) for t, c in zip(inputs[0], inputs[1])],  # target
                    add_special_tokens=True,  # Add [CLS] and [SEP]
                    return_tensors='pt',  # Return PyTorch tensors
                    padding='longest',  # Pad to the maximum length
                    max_length=512
                )
            else:
                tokenized_inputs = tokenizer(inputs[0], padding='longest', return_tensors='pt')
            tokenized_inputs = tokenized_inputs.to(device)
            labels = labels.to(device)
            outputs = model(**tokenized_inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels)
            all_predictions.extend(predicted)

    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(all_predictions, all_labels)

    return val_accuracy, val_precision, val_recall, val_f1