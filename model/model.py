import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaForToxicClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_class):
        super(RobertaForToxicClassification, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.classification_head = nn.Linear(self.roberta.config.hidden_size, num_class)

    def forward(self, input_ids,attention_mask):
        print(" TOKENIZED INPUTS ", input_ids.shape, " ",attention_mask.shape)
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = roberta_outputs.pooler_output
        output = self.classification_head(pooler_output)
        return output
