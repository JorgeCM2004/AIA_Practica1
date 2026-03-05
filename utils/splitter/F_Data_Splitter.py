import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class Data_Splitter:
	def split(self, seed=None, test_size=0.2):
		if seed and isinstance(seed, int):
			random.seed(seed)
		data_folder = Path(__file__).parent.parent.parent / "data"
		if (data_folder / "fetal_health.csv").exists():
			(data_folder / "fetal_health_train.csv").unlink(missing_ok=True)
			(data_folder / "fetal_health_test.csv").unlink(missing_ok=True)
			train_df, test_df = train_test_split(
				pd.read_csv((data_folder / "fetal_health.csv").resolve()),
				test_size=test_size,
				random_state=seed,
			)
			train_df.to_csv(
				(data_folder / "fetal_health_train.csv").resolve(), index=False
			)
			test_df.to_csv(
				(data_folder / "fetal_health_test.csv").resolve(), index=False
			)
			return train_df, test_df
		else:
			raise ValueError("Debes descargar el dataset primero.")
