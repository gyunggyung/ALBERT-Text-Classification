# Text Classification, [🇰🇷](ko.md) 버전

ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation. 

For a technical description of the algorithm, see our paper: [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

Using the ktrain library, proceed with the text classification. Detailed descriptions can be found at [Blog](https://hipgyung.tistory.com/93)


## System requirements
``` python
pip install requirements.txt
```

## How to use
With simple commands, you can proceed with text classification for datasets made up of csv files, use `main.py`:
```
python main.py \
	--csv data.csv \
	--label Category \
	--data Resume --epoch 5
```
### My case
```
python main.py \
	--csv data.csv \
	--label label_name \
	--data data_name \
	--epoch 5
```

## ☄️ Available models
Replace the bottom part with the model you want.
``` python
	MODEL_NAME = 'albert-base-v2'
```

| Model  | Type of detail  |
|----------|------------------------------|
| BERT: |*bert-base-uncased, bert-large-uncased, bert-base-multilingual-uncased, and others.*|
| DistilBERT: |*distilbert-base-uncased, distilbert-base-multilingual-cased, distilbert-base-german-cased, and others*|
| ALBERT: |*albert-base-v2, albert-large-v2, and others*|
| RoBERTa: |*roberta-base, roberta-large, roberta-large-mnli*|
| XLM: |*xlm-mlm-xnli15–1024, xlm-mlm-100–1280, and others*|
| XLNet: | *xlnet-base-cased, xlnet-large-cased*|


## Outstanding performance
![](img.png)  
#### 📈 97

## predictor
You can use the function below.
``` python
def predictor(learner, test):
	predictor = ktrain.get_predictor(learner.model, preproc=t)
	print(predictor.predict(test))

```

## tensorboard
```
tensorboard \
	--logdir==training:your_log_dir \
	--host=127.0.0.1
```
### Example
```
tensorboard \
	--logdir==training:logs/ \
	--host=127.0.0.1
```

## 🔬 Library
> https://github.com/amaiya/ktrain