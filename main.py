import pandas as pd
from sklearn.model_selection import train_test_split
import ktrain
from ktrain import text
import argparse

import os

import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--csv', help='train model csv file')
parser.add_argument('--label', help='train label of dataset')
parser.add_argument('--data', help='train dataset')
parser.add_argument('--epoch', help='traing Epoch')

args = parser.parse_args()

def read_dataset(dataset, data, label):
	df = pd.read_csv(dataset)
	label_list = list(set(df["Category"]))
	df.sample(frac=1)
	x_train, x_test, y_train, y_test = train_test_split(
    	list(df[data]), list(df[label]), test_size=0.33, random_state=42)
	return x_train, x_test, y_train, y_test, label_list

def train_model(x_train, x_test, y_train, y_test, label_list, epoch, checkpoint_path):
	MODEL_NAME = 'albert-base-v2'
	t = text.Transformer(MODEL_NAME, maxlen=500, class_names=label_list)
	trn = t.preprocess_train(x_train, y_train)
	val = t.preprocess_test(x_test, y_test)
	model = t.get_classifier()
	learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
	learner.fit_onecycle(3e-5, int(epoch),checkpoint_folder = checkpoint_path)
	return learner, model

def predictor(learner, test):
	predictor = ktrain.get_predictor(learner.model, preproc=t)
	print(predictor.predict(test))

if __name__ == "__main__":
	x_train, x_test, y_train, y_test, label_list = read_dataset(args.csv, args.data, args.label)

	checkpoint_path = "training_1/cp.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)

	learner, model = train_model(x_train, x_test, y_train, y_test, label_list, int(args.epoch), checkpoint_path)
	model.summary()