import torch
import pandas as pd
from typing import List, Tuple
from transformers import BertTokenizer

def read_csv(
    path: str
) -> pd.DataFrame:
    """
    Reads the csv file.

    Args:
        path (str): the file path.

    Returns:
        pd.DataFrame: the dataframe.
    """
    return pd.read_csv(path, sep=",")

def bert_preprocessing(
    texts: List,
    max_len: int,
    tokenizer: BertTokenizer
) -> Tuple[List, List]:
    """
    Preprocessing the fexts to be used with BERT (using BERT Tokenizer).

    Args:
        texts (List): a list of the texts to be processed.
        max_len (int): the max length of the text.
        tokenizer (BertTokenizer): the BERT tokenizer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the input ids and attention masks obtained.
    """
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_text = tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids.append(encoded_text.get("input_ids"))
        attention_masks.append(encoded_text.get("attention_mask"))
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks