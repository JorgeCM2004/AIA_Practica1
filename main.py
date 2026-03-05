from utils import Data_Downloader

SEED = 42


def main():
	downloader = Data_Downloader()
	downloader.download()


if __name__ == "__main__":
	main()
