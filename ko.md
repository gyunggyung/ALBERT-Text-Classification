# ALBERT-Text-Classification (국민청원 카테고리 분류)
 랜덤하게 국민청원 3,000개를 뽑아서 find-tuning 하는 모델입니다. 코드에 대한 자세한 설명은 [Blog](https://hipgyung.tistory.com/93)에 명시되어 있습니다.

## 패키지 설치
``` python
pip install ktrain
```

## 사용 방법
```
python main.py --csv test_data.csv --label category --data petition_overview --epoch 5

python main.py --csv data.csv --label label_name --data data_name --epoch 5

python main.py --csv petition_data_all.csv --label category --data petition_overview --epoch 30
```

## 사용 가능한 모델들
- BERT: bert-base-uncased, bert-large-uncased, bert-base-multilingual-uncased, and others.
- DistilBERT: distilbert-base-uncased, distilbert-base-multilingual-cased, distilbert-base-german-cased, and others
- ALBERT: albert-base-v2, albert-large-v2, and others
- RoBERTa: roberta-base, roberta-large, roberta-large-mnli
- XLM: xlm-mlm-xnli15–1024, xlm-mlm-100–1280, and others
- XLNet: xlnet-base-cased, xlnet-large-cased

## 결과 성능
![]()  


## predictor
아래 함수를 이용해서 직접 예측이 가능합니다.
``` python
def predictor(learner, test):
	predictor = ktrain.get_predictor(learner.model, preproc=t)
	print(predictor.predict(test))

```

## tensorboard
```
pip install tensorboard==1.12.2

#tensorboard --logdir==training:your_log_dir --host=127.0.0.1
tensorboard --logdir==training:logs/ --host=127.0.0.1
```

## Library
> https://github.com/amaiya/ktrain