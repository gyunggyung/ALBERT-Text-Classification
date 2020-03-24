import pandas as pd
from sklearn.model_selection import train_test_split
import ktrain
from ktrain import text

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

if __name__ == "__main__":
	x_train, x_test, y_train, y_test = read_dataset("data.csv", "Resume", "Category")
	learner = train_model(x_train, x_test, y_train, y_test)
	
	predictor = ktrain.get_predictor(learner.model, preproc=t)

	test = """Compensation and Benefits Resume Example
	A compensation and benefits specialist manages employee compensation, annual performance reviews and employee benefits. In a specialist role, the employee does additional administrative tasks, although management implements strategic planning in the company.

	To build an impressive compensation and benefits resume, you must prove that you are extremely knowledgeable about benefits packages and compensation. The more HR knowledge you possess generally, the better you will be able to answer specific employees’ questions, no matter how complex they may be.

	Take it a step further and share your administrative background; doing so will show hiring managers that you are thorough and do not let errors slip past.

	"""

	print(predictor.predict(test))