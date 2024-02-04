import torch
from transformers import BertConfig, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import BertTokenizer
from flask import Flask, request
from src.preprocessing import preprocessing
from src.utils import read_csv, bert_preprocessing
from src.bert import BERT
from src.dataset import create_dataloader
from sklearn.metrics import f1_score
from typing import Tuple

app = Flask(__name__)

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    scheduler: get_linear_schedule_with_warmup
) -> Tuple[float, float]:
    """
    Function responsible for the training step.
    
    Args:
        device (torch.device): the device (cpu or cuda).
        dataloader (torch.utils.data.DataLoader): the training dataloader.
        optimizer (torch.nn.optim): the optimizer that will be used.
        model (torch.nn.Module): the Bert model.
    Returns:
        Tuple[float, float]: current epoch training loss and accuracy.
    """
    model.train()
    train_loss = 0
    train_f1 = 0
    
    for batch in dataloader:
        input_id, attention_mask, target = batch
        input_id, attention_mask, target = input_id.to(device), attention_mask.to(device), target.to(device)
        target = target.reshape((target.shape[0], 1))
        target = target.float()

        optimizer.zero_grad()
        loss, logits = model(input_id, attention_mask, target)
        
        train_loss += loss.item()
        loss.backward()

        optimizer.step()

        scheduler.step()

        prediction = torch.argmax(logits, axis=1).flatten()
        target = target.flatten()

        prediction = (logits > 0.5) * 1.
        train_f1 += f1_score(
            target.detach().cpu().data.numpy(),
            prediction.detach().cpu().numpy(),
            average="weighted"
        )

    train_loss /= len(dataloader)
    train_f1 /= len(dataloader)
    return train_f1, train_loss

def test(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader
):
    """
    Function responsible for the test step.
    
    Args:
        device (torch.device): the device (cpu or cuda).
        validation_loader (torch.utils.data.DataLoader): the validation dataloader.
        model (torch.nn.Module): the Bert model.
    
    Returns:
        Tuple[float, float]: current epoch test loss and accuracy.
    """
    model.eval()
    test_loss = 0
    test_f1 = 0
    
    with torch.inference_mode():
        for batch in dataloader:
            input_id, attention_mask, target = batch
            input_id, attention_mask, target = input_id.to(device), attention_mask.to(device), target.to(device)
            target = target.reshape((target.shape[0], 1))
            target = target.float()
            loss, logits = model(input_id, attention_mask, target)
            
            test_loss += loss.item()
            prediction = torch.argmax(logits, axis=1).flatten()
            target = target.flatten()

            prediction = (logits > 0.5) * 1.
            test_f1 += f1_score(
                target.detach().cpu().data.numpy(),
                prediction.detach().cpu().numpy(),
                average="weighted"
            )
        
    test_loss /= len(dataloader)
    test_f1 /= len(dataloader)
    return test_f1, test_loss

@app.route("/train", methods=["GET"])
def train_model():
    epochs = 20
    batch_size = 16
    max_len = 70
    lr = 5e-5
    cv = 5

    # reading and cleaning the dataset
    train_df = read_csv("./data/train.csv")
    train_df = train_df.drop_duplicates(subset=["id"], keep=False)
    train_df = train_df.reset_index(drop=True)
    train_df = train_df.drop(columns=["id", "keyword", "location"])
    train_df["cleaned_text"] = train_df["text"].apply(preprocessing)
    train_df = train_df.drop_duplicates()
    train_df = train_df.dropna()
    train_df = train_df.reset_index(drop=True)
    
    # creating the Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True
    )
    
    # preprocessing the texts to be exactly as Bert needs
    X, X2 = bert_preprocessing(
        texts=train_df["cleaned_text"].values.tolist(),
        max_len=max_len,
        tokenizer=tokenizer
    )

    y = torch.tensor(train_df["target"].values.tolist())

    # creating the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT().to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        eps=1e-08
    )

    # splitting the training data into train and validation
    # using the stratified suffled cross validation method
    stratified_split = StratifiedShuffleSplit(
        n_splits=cv,
        test_size=0.2
    )

    for fold, (train_index, test_index) in enumerate(stratified_split.split(X, y)):
        print(f"\n---------- FOLD {fold+1} ----------")
        input_ids_train, att_masks_train = X[train_index, :], X2[train_index, :]
        input_ids_test, att_masks_test = X[test_index, :], X2[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        
        # creating the dataloader
        train_dataloader = create_dataloader(
            input_ids=input_ids_train,
            attention_masks=att_masks_train,
            labels=y_train,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = 0,
            num_training_steps = len(train_dataloader) * epochs
        )
        
        test_dataloader = create_dataloader(
            input_ids=input_ids_test,
            attention_masks=att_masks_test,
            labels=y_test,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False
        )

        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}")
            train_f1, train_loss = train(
                model=model,
                optimizer=optimizer,
                device=device,
                dataloader=train_dataloader,
                scheduler=scheduler
            )
            
            test_f1, test_loss = test(
                model=model,
                device=device,
                dataloader=test_dataloader
            )

            print(train_f1, train_loss)

if __name__ == "__main__":
    app.run(debug=True)