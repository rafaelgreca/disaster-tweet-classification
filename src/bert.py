import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from typing import Tuple

class BERT(nn.Module):
    def __init__(
        self,
        freeze: bool = False
    ) -> None:
        super(BERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 1,
            output_attentions = False,
            output_hidden_states = False
        )
        
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = True
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=None,
            attention_mask=attention_masks,
            labels=target,
            return_dict=None
        )
        return output["loss"], output["logits"]