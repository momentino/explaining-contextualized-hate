import torch
from utils.utils import calculate_metrics

def eval(model, tokenizer, val_loader):
    model.eval()

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            """ 
                We need to consider two separate cases: the one where the context is absent and the one where it is present.
                When we have the context, we need to concatenate them together and we use the special method encode_plus
            """
            if (len(inputs) == 0):
                tokenized_inputs = tokenizer.encode_plus(
                    inputs[0],  # target
                    inputs[1],  # context
                    add_special_tokens=True,  # Add [CLS] and [SEP]
                    return_tensors='pt',  # Return PyTorch tensors
                    padding='longest',  # Pad to the maximum length
                )
            else:
                tokenized_inputs = tokenizer(inputs[0], padding='longest', return_tensors='pt')
            outputs = model(**tokenized_inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels)
            all_predictions.extend(predicted)

    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(all_predictions, all_labels)

    return {
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1_score': val_f1
    }