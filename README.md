# ALBERT-Text-Classification
Detailed descriptions can be found at [Blog](https://hipgyung.tistory.com/93)

## How to use
```
pip install ktrain

python main.py
```

## Available models
- BERT: bert-base-uncased, bert-large-uncased, bert-base-multilingual-uncased, and others.
- DistilBERT: distilbert-base-uncased, distilbert-base-multilingual-cased, distilbert-base-german-cased, and others
- ALBERT: albert-base-v2, albert-large-v2, and others
- RoBERTa: roberta-base, roberta-large, roberta-large-mnli
- XLM: xlm-mlm-xnli15–1024, xlm-mlm-100–1280, and others
- XLNet: xlnet-base-cased, xlnet-large-cased

## Parts to be modified
```
	x_train, x_test, y_train, y_test = read_dataset("data.csv", "Resume", "Category") #dataset name, dataset, label
```