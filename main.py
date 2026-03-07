from utils import Data_Downloader, Data_Preprocessor, Data_Splitter, Model_Manager

SEED = 42  # None para no reproducibilidad.


def main():
	downloader = Data_Downloader()
	downloader.download()
	splitter = Data_Splitter()
	train, test = splitter.split(SEED)
	preprocessor = Data_Preprocessor()
	x_train, y_train = preprocessor.preprocess(
		train, training_dataset=True, target="fetal_health"
	)
	x_test, y_test = preprocessor.preprocess(
		test, training_dataset=False, target="fetal_health"
	)
	model_manager = Model_Manager(seed=SEED)
	model_manager.train(x_train, y_train)
	models, y_hats = model_manager.predict(x_test)
	for model, y_hat in zip(models, y_hats):
		pass


if __name__ == "__main__":
	main()
