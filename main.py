from utils import Data_Downloader, Data_Preprocessor, Data_Splitter

SEED = 42


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


if __name__ == "__main__":
	main()
