import pandas as pd
from sklearn.model_selection import train_test_split
import ktrain
from ktrain import text
import argparse

def read_dataset(dataset, data, label):
	df = pd.read_csv(dataset)
	label_list = list(set(df["Category"]))
	df.sample(frac=1)
	x_train, x_test, y_train, y_test = train_test_split(
    	list(df[data]), list(df[label]), test_size=0.33, random_state=42)
	return x_train, x_test, y_train, y_test


def train_model(x_train, x_test, y_train, y_test):
	MODEL_NAME = 'albert-base-v2'
	t = text.Transformer(MODEL_NAME, maxlen=500, class_names=label_list)
	trn = t.preprocess_train(x_train, y_train)
	val = t.preprocess_test(x_test, y_test)
	model = t.get_classifier()
	learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
	learner.fit_onecycle(3e-5, 5)
	return learner

def parser():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--csv', metavar='N', type='str',
	                    help='train model csv file')
	parser.add_argument('--label', metavar='N', type='str',
	                    help='train label of dataset')
	parser.add_argument('--data', metavar='N', type='str',
	                    help='train dataset')

	return parser.parse_args()

def predictor(learner, test):
	predictor = ktrain.get_predictor(learner.model, preproc=t)
	print(predictor.predict(test))

if __name__ == "__main__":
	args = parser()
	x_train, x_test, y_train, y_test = read_dataset(args.csv, args.data, args.label)
	learner = train_model(x_train, x_test, y_train, y_test)