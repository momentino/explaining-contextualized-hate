import torch
import copy

from utils.utils import calculate_metrics
from eval.eval import eval
def train(model, tokenizer, train_loader, val_loader, num_epochs, optimizer, criterion, model_save_path):

    best_val_f1 = 0.0

    all_predictions = []
    all_labels = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0


        for i, (inputs, labels) in enumerate(train_loader):
            """ 
                We need to consider two separate cases: the one where the context is absent and the one where it is present.
                When we have the context, we need to concatenate them together and we use the special method encode_plus
            """
            if(len(inputs)== 0):
                tokenized_inputs = tokenizer.encode_plus(
                    inputs[0], # target
                    inputs[1], #context
                    add_special_tokens=True,  # Add [CLS] and [SEP]
                    return_tensors='pt',  # Return PyTorch tensors
                    padding='longest',  # Pad to the maximum length
                )
            else:
                tokenized_inputs = tokenizer(inputs[0], padding='longest', return_tensors='pt')
            outputs = model(**tokenized_inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels)
            all_predictions.extend(predicted)

        train_loss = running_loss / len(train_loader)

        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(all_predictions, all_labels)

        """ Evaluate training """
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {train_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.2f}%'
              f'Training Precision: {train_precision:.2f}%'
              f'Training Recall: {train_recall:.2f}%'
              f'Training F1: {train_f1:.2f}%')
        print(" Wait for training...")
        """ Validation """
        val_accuracy, val_precision, val_recall, val_f1 = eval(model, val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {train_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.2f}%')

        # Save the model if validation F1 improves
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved with validation F1: {val_f1:.4f}")