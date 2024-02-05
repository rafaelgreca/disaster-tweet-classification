import torch
import os
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import BertTokenizer
from flask import Flask, jsonify, request
from src.preprocessing import preprocessing
from src.utils import read_csv, bert_preprocessing
from src.bert import BERT, SaveBestModel
from src.dataset import create_dataloader
from sklearn.metrics import f1_score
from typing import Tuple

# defining global variables
epochs = 7
batch_size = 32
max_len = 70
lr = 2e-5
cv = 5
app = Flask(__name__)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    scheduler: get_linear_schedule_with_warmup,
) -> Tuple[float, float]:
    """
    Function responsible for the training step.

    Args:
        model (torch.nn.Module): the Bert model.
        optimizer (torch.nn.optim.AdamW): the optimizer that will be used.
        device (torch.device): the torch device (cpu or cuda).
        dataloader (torch.utils.data.DataLoader): the training dataloader.
        scheduler (get_linear_schedule_with_warmup): BERT's warmup scheduler.

    Returns:
        Tuple[float, float]: current epoch training f1 score and loss, respectively.
    """
    model.train()
    train_loss = 0
    train_f1 = 0

    for batch in dataloader:
        input_id, attention_mask, target = batch
        input_id, attention_mask, target = (
            input_id.to(device),
            attention_mask.to(device),
            target.to(device),
        )
        target = target.reshape((target.shape[0], 1))
        target = target.float()

        optimizer.zero_grad()
        loss, logits = model(input_id, attention_mask, target)

        train_loss += loss.item()
        loss.backward()

        optimizer.step()

        scheduler.step()

        # because this is a binary classification, we get the index of
        # the maximum logit value
        prediction = torch.argmax(logits, axis=1).flatten()
        target = target.flatten()
        prediction = (logits > 0.5) * 1.0

        train_f1 += f1_score(
            target.detach().cpu().data.numpy(),
            prediction.detach().cpu().numpy(),
            average="weighted",
        )

    train_loss /= len(dataloader)
    train_f1 /= len(dataloader)
    return train_f1, train_loss


def test(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
):
    """
    Function responsible for the test/validation step.

    Args:
        model (torch.nn.Module): the Bert model.
        device (torch.device): the torch device (cpu or cuda).
        dataloader (torch.utils.data.DataLoader): the validation dataloader.

    Returns:
        Tuple[float, float]: current epoch validation f1 score and loss, respectively.
    """
    model.eval()
    test_loss = 0
    test_f1 = 0

    with torch.inference_mode():
        for batch in dataloader:
            input_id, attention_mask, target = batch
            input_id, attention_mask, target = (
                input_id.to(device),
                attention_mask.to(device),
                target.to(device),
            )
            target = target.reshape((target.shape[0], 1))
            target = target.float()
            loss, logits = model(input_id, attention_mask, target)

            test_loss += loss.item()

            # because this is a binary classification, we get the index of
            # the maximum logit value
            prediction = torch.argmax(logits, axis=1).flatten()
            target = target.flatten()

            prediction = (logits > 0.5) * 1.0
            test_f1 += f1_score(
                target.detach().cpu().data.numpy(),
                prediction.detach().cpu().numpy(),
                average="weighted",
            )

    test_loss /= len(dataloader)
    test_f1 /= len(dataloader)
    return test_f1, test_loss


def inference(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
):
    """
    Function responsible for the inference step.

    Args:
        model (torch.nn.Module): the Bert model.
        device (torch.device): the torch device (cpu or cuda).
        dataloader (torch.utils.data.DataLoader): the test dataloader.

    Returns:
        torch.Tensor: the data predictions.
    """
    model.eval()
    predictions = []

    with torch.inference_mode():
        for batch in dataloader:
            input_id, attention_mask = batch
            input_id, attention_mask = input_id.to(device), attention_mask.to(device)

            logits = model(input_id, attention_mask, None)

            # because this is a binary classification, we get the index of
            # the maximum logit value
            prediction = torch.argmax(logits, axis=1).flatten()
            prediction = (logits > 0.5) * 1.0
            prediction = prediction.flatten()
            prediction = prediction.to(dtype=torch.int)

            # changing the prediction tensor's device to cpu
            if prediction.get_device() != "cpu":
                prediction = prediction.cpu()

            predictions.extend(prediction.detach().numpy().tolist())

    return predictions


@app.route("/inference", methods=["POST"])
def inference_model():
    """
    Endpoint used to inference a trained model on the test set.

    Args:
        data (json): a json containing the model that will be used.
            E.g.: {
                "model_name": "bert_0"
            }

    Returns:
        data (json): a json containing the prediction for each
            text in the test data.
    """
    data = request.json
    model_name = data["model_name"]

    try:
        assert "model_name" in data.keys()
    except AssertionError:
        return "Please pass the model's name using the 'model_name' parameter\n"

    try:
        # reading and cleaning the dataset
        test_df = read_csv("./data/test.csv")
        test_df = test_df.drop(columns=["keyword", "location"])
        test_df["cleaned_text"] = test_df["text"].apply(preprocessing)
    except FileNotFoundError:
        return (
            "Please create a folder named 'data' on the folder root "
            "and then put the 'test.csv' in it.\n"
        )

    # creating the Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # preprocessing the texts to be exactly as Bert needs
    input_ids, att_masks = bert_preprocessing(
        texts=test_df["cleaned_text"].values.tolist(),
        max_len=max_len,
        tokenizer=tokenizer,
    )

    # creating the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT().to(device)

    try:
        # loading the trained model parameters
        model.load_state_dict(
            torch.load(os.path.join(os.getcwd(), "models", f"{model_name}.pth"))[
                "model_state_dict"
            ]
        )
    except FileNotFoundError:
        return f"{os.path.join(os.getcwd(), 'models', f'{model_name}.pth')} not found\n"

    # creating the test dataloader
    test_dataloader = create_dataloader(
        input_ids=input_ids,
        attention_masks=att_masks,
        labels=None,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )

    # predicting the labels
    prediciton = inference(model=model, device=device, dataloader=test_dataloader)

    # transforming the dataset into dict and then to json
    test_df["prediction"] = prediciton
    test_df = test_df[["id", "text", "cleaned_text", "prediction"]]
    inferece_output = test_df.to_dict(orient="records")

    return jsonify(inferece_output)


@app.route("/train", methods=["GET"])
def train_model():
    """
    Endpoint used to train the model.

    Returns:
        data (json): a json containing the training and validation
            metrics for each cross validation's fold.
    """
    try:
        # reading and cleaning the dataset
        train_df = read_csv("./data/train.csv")
        train_df = train_df.drop_duplicates(subset=["id"], keep=False)
        train_df = train_df.reset_index(drop=True)
        train_df = train_df.drop(columns=["id", "keyword", "location"])
        train_df["cleaned_text"] = train_df["text"].apply(preprocessing)
        train_df = train_df.drop_duplicates()
        train_df = train_df.dropna()
        train_df = train_df.reset_index(drop=True)
    except FileNotFoundError:
        return (
            "Please create a folder named 'data' on the folder root "
            "and then put the 'train.csv' in it.\n"
        )

    # creating the Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # preprocessing the texts to be exactly as Bert needs
    X, X2 = bert_preprocessing(
        texts=train_df["cleaned_text"].values.tolist(),
        max_len=max_len,
        tokenizer=tokenizer,
    )

    y = torch.tensor(train_df["target"].values.tolist())

    # creating the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT().to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, eps=1e-08)

    # splitting the training data into train and validation
    # using the stratified suffled cross validation method
    stratified_split = StratifiedShuffleSplit(n_splits=cv, test_size=0.2)

    # creating the model checkpoint object
    sbm = SaveBestModel(output_dir=os.path.join(os.getcwd(), "models"))

    training_output = {}

    for fold, (train_index, test_index) in enumerate(stratified_split.split(X, y)):
        print(f"\n---------- FOLD {fold+1} ----------")
        input_ids_train, att_masks_train = X[train_index, :], X2[train_index, :]
        input_ids_test, att_masks_test = X[test_index, :], X2[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        # creating the training dataloader
        train_dataloader = create_dataloader(
            input_ids=input_ids_train,
            attention_masks=att_masks_train,
            labels=y_train,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )

        # creating the scheduler warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * epochs,
        )

        # creating the test dataloader
        test_dataloader = create_dataloader(
            input_ids=input_ids_test,
            attention_masks=att_masks_test,
            labels=y_test,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
        )

        # training loop
        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}")
            train_f1, train_loss = train(
                model=model,
                optimizer=optimizer,
                device=device,
                dataloader=train_dataloader,
                scheduler=scheduler,
            )

            test_f1, test_loss = test(
                model=model, device=device, dataloader=test_dataloader
            )

            # saving the best model
            sbm(
                current_valid_f1=test_f1,
                current_valid_loss=test_loss,
                current_train_f1=train_f1,
                current_train_loss=train_loss,
                epoch=epoch,
                fold=fold,
                optimizer=optimizer,
                model=model,
            )

        # saving the best model's metrics for that fold
        training_output[f"{fold}"] = {
            "train_f1": f"{sbm.best_train_f1}",
            "train_loss": f"{sbm.best_train_loss}",
            "valid_f1": f"{sbm.best_valid_f1}",
            "valid_loss": f"{sbm.best_valid_loss}",
        }

    return jsonify(training_output)


if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
