import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List

def create_dataloader(
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    num_workers: int,
    shuffle: bool
) -> DataLoader:
    """
    Creates a dataloader to be used for BERT.

    Args:
        input_ids (torch.Tensor): the input ids obtained from BERT tokenizer.
        attention_masks (torch.Tensor): the attention mask obtained from BERT tokenizer.
        labels (torch.Tensor): the texts labels.
        batch_size (int): the batch size.
        num_workers (int): the number of workers.
        shuffle (bool): shuffle or not shuffle the data.

    Returns:
        DataLoader: the dataloader created.
    """
    dataset = TensorDataset(
        input_ids, 
        attention_masks,
        labels
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return dataloader