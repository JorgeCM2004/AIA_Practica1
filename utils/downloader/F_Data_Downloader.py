from pathlib import Path

import kagglehub


class Data_Downloader:
	def download(self):
		kagglehub.dataset_download(
			"andrewmvd/fetal-health-classification",
			force_download=True,
			output_dir=Path(__file__).parent.parent.parent / "data",
		)
