# Disaster Tweet Classification

This project aims to create a Deep Learning model designed to tackle [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/code?competitionId=17777) competition proposed by Kaggle. However, it's not focused on training and evaluating different algorithms to see which has a better performance on Kaggle's ranking, but rather on building an application using Flask and Docker to use the training dataset to build and train a BERT model and then use it to make predictions on the test dataset.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install this package, first clone the repository to the directory of your choice using the following command:
```bash
git clone https://github.com/rafaelgreca/disaster-tweet-classification.git
```

Finally, you need to create a conda or virtual environment and install the requirements. This can be done using the following command:

For `pip` use the following command:
```bash
conda create --name disaster-classification python=3.11
conda activate disaster-classification
pip install -r requirements.txt
```

## Getting Started

### Download the Dataset

Before continuing, to the code work properly you need to download the dataset correctly. If you install using other sources, the code might not work. Download the dataset using [Kaggle's link](https://www.kaggle.com/competitions/nlp-getting-started/code?competitionId=17777). After downloading it, create a `data` folder **on the root** and put the `train.csv` and `test.csv` files inside of it.

### Directory Structure

```bash
./
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── bert.py
│   ├── dataset.py
│   ├── preprocessing.py
│   └── utils.py
├── __init__.py
├── LICENSE
├── README.md
├── requirements.txt
├── Dockerfile
└── api.py
```

Explaining briefly the main folders and files:

- `requirements.txt`: the main libraries used to develop the project;
- `src`: where the core functions are implemented, such as the text preprocessing steps (`preprocessing.py`), BERT model definition (`bert.py`), the dataset creation (`dataset.py`), and input/output files' operations (`utils.py`);
- `api.py`: the main file responsible for creating the API's endpoints (training and inference). Additionally, also all functions that were used in both endpoints.

### Running the Code

Building the Docker image:

```bash
sudo docker build -f Dockerfile -t disaster-tweet . --no-cache
```

Running the Docker container:

```bash
sudo docker run -d -p 8000:5000 --name disaster disaster-tweet
```

Training the BERT model (we are using cross-validation with 5 folds, therefore 5 BERT models will be created and trained, then saved in a folder called `models` located on the root folder):

```bash
curl -X GET http://127.0.0.1:8000/train
```

An example of what the API will return after the BERT is trained:

```json
{
  "0": {
    "train_f1": "0.8549920922517874",
    "train_loss": "0.12686705062205486",
    "valid_f1": "0.8109330292567293",
    "valid_loss": "0.14864667398311818"
  },
  "1": {
    "train_f1": "0.8335076047278006",
    "train_loss": "0.14481979079465282",
    "valid_f1": "0.8793904647160197",
    "valid_loss": "0.09332963859196752"
  },
  "2": {
    "train_f1": "0.8962081752840937",
    "train_loss": "0.10410643358060971",
    "valid_f1": "0.9332780567691313",
    "valid_loss": "0.05860341369384514"
  },
  "3": {
    "train_f1": "0.9299154303900699",
    "train_loss": "0.07443954636580608",
    "valid_f1": "0.942967428795232",
    "valid_loss": "0.053271645408434175"
  },
  "4": {
    "train_f1": "0.9423206835786697",
    "train_loss": "0.06326480431759611",
    "valid_f1": "0.9519999548124859",
    "valid_loss": "0.042643080713655"
  }
}
```

Using the trained model on the test dataset (the model's name can be `bert_fold0`, `bert_fold1`, `bert_fold2`, `bert_fold3` or `bert_fold4`):

```bash
curl -X POST http://127.0.0.1:8000/inference -H \ 
'Content-Type: application/json' -H 'Accept: application/json' \
-d '{"model_name": "bert_fold0"}'
```

A small sample of what the API will return after the inference:

```json
[
  {
    "cleaned_text": "death toll suicide car bombing pg position village rajman eastern province hasaka risen",
    "id": 10858,
    "prediction": 1,
    "text": "The death toll in a #IS-suicide car bombing on a #YPG position in the Village of Rajman in the eastern province of Hasaka has risen to 9"
  },
  {
    "cleaned_text": "earthquake safety los angeles uo safety fasteners xrwn",
    "id": 10861,
    "prediction": 1,
    "text": "EARTHQUAKE SAFETY LOS ANGELES \u0089\u00db\u00d2 SAFETY FASTENERS XrWn"
  },
  {
    "cleaned_text": "storm ri worse last hurricane city amp others hardest hit yard looks like bombed around still without power",
    "id": 10865,
    "prediction": 1,
    "text": "Storm in RI worse than last hurricane. My city&amp;3others hardest hit. My yard looks like it was bombed. Around 20000K still without power"
  },
  {
    "cleaned_text": "green line derailment chicago",
    "id": 10868,
    "prediction": 0,
    "text": "Green Line derailment in Chicago http://t.co/UtbXLcBIuY"
  },
  {
    "cleaned_text": "meg issues hazardous weather outlook hwo",
    "id": 10874,
    "prediction": 1,
    "text": "MEG issues Hazardous Weather Outlook (HWO) http://t.co/3X6RBQJHn3"
  },
  {
    "cleaned_text": "city calgary activated municipal emergency plan yy storm",
    "id": 10875,
    "prediction": 1,
    "text": "#CityofCalgary has activated its Municipal Emergency Plan. #yycstorm"
  }
]
```

## Contributing

Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. Don't forget to give the project a star! Thanks again!

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Author: Rafael Greca Vieira - [GitHub](github.com/rafaelgreca/) - [LinkedIn](https://www.linkedin.com/in/rafaelgreca/) - rgvieira97@gmail.com
