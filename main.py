from utils import Data_Downloader, Data_Splitter

SEED = 42


def main():
	downloader = Data_Downloader()
	downloader.download()
	splitter = Data_Splitter()
	splitter.split(SEED)


if __name__ == "__main__":
	main()
