import torch
import torch.nn as nn
import numpy as np
import os
from transformers import BertForSequenceClassification
from typing import Tuple

def weight_init(m: torch.nn.Module):
    """
    Initalize all the weights in the PyTorch model to be the same as Keras.

    All credits to:
    https://discuss.pytorch.org/t/same-implementation-different-results-between-keras-and-pytorch-lstm/39146
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.zeros_(m.bias_ih_l0)
        nn.init.zeros_(m.bias_hh_l0)

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Args:
            output_dir (str): the output folder directory.
            model_name (str): the model's name.
            dataset (str): which dataset is being used (coraa, emodb or ravdess).
        """
        self.best_valid_loss = float(np.Inf)
        self.best_valid_f1 = float(np.NINF)
        self.best_train_f1 = float(np.NINF)
        self.best_train_loss = float(np.Inf)
        self.output_dir = output_dir
        self.save_model = False
        self.best_epoch = -1
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(
        self,
        current_valid_loss: float,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim,
        fold: int,
        current_valid_f1: float = None,
        current_train_f1: float = None,
        current_train_loss: float = None,
    ) -> None:
        """
        Saves the best trained model.

        Args:
            current_valid_loss (float): the current validation loss value.
            current_valid_f1 (float): the current validation f1 score value.
            current_test_f1 (float): the current test f1 score value.
            current_train_f1 (float): the current train f1 score value.
            epoch (int): the current epoch.
            model (nn.Module): the trained model.
            optimizer (torch.optim): the optimizer objet.
            fold (int): the current fold.
        """
        if current_valid_f1 > self.best_valid_f1:
            self.best_valid_loss = current_valid_loss
            self.best_valid_f1 = current_valid_f1
            self.best_train_f1 = current_train_f1
            self.best_train_loss = current_train_loss
            self.best_epoch = epoch
            self.save_model = True

        if self.save_model:
            self.print_summary()

            if not fold is None:
                path = os.path.join(
                    self.output_dir, f"bert_fold{fold}.pth"
                )
            else:
                path = os.path.join(self.output_dir, "bert.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            self.save_model = False

    def print_summary(self) -> None:
        """
        Print the best model's metric summary.
        """
        print("\nSaving model...")
        print(f"Epoch: {self.best_epoch}")
        print(f"Train F1-Score: {self.best_train_f1:1.6f}")
        print(f"Train Loss: {self.best_train_loss:1.6f}")
        print(f"Validation F1-Score: {self.best_valid_f1:1.6f}")
        print(f"Validation Loss: {self.best_valid_loss:1.6f}\n")

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
            
        self.bert.apply(weight_init)
            
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