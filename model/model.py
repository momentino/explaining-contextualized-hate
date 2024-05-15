import torch
import torch.nn as nn
from transformers import RobertaModel

class RoBERTaWrapper(nn.Module):
    def __init__(self, pretrained_model_name):
        super(RoBERTaWrapper, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)

    def forward(self, input):
        outputs = self.roberta(input_ids=input['input_ids'], attention_mask=input['attention_mask'])
        return outputs
