import pandas as pd
from sklearn.preprocessing import StandardScaler


class Data_Preprocessor:
	def __init__(self):
		self.std_scaler: StandardScaler = None

	def preprocess(
		self,
		data: pd.DataFrame,
		training_dataset: bool,
		target: str = "fetal_health",
		columns=None,
	):
		self.check_nulls(data)
		x, labels = self.separate(data, target)
		x = self.standardize(x, training_dataset, columns)
		return x, labels

	def check_nulls(self, data: pd.DataFrame):
		number_of_nulls = data.isnull().sum().sum()
		print(
			"​✅ Dataset sin valores nulos."
			if number_of_nulls <= 0
			else "⚠️ Dataset con valores no válidos, es necesario limpieza."
		)

	def separate(self, data: pd.DataFrame, target: str):
		if target not in data.columns:
			raise ValueError("Target debe ser una de las columnas del dataset.")
		labels = data[target]
		x = data.drop(columns=target)
		return x, labels

	def standardize(self, x: pd.DataFrame, training_dataset: bool, cols=None):
		if not cols:
			cols = x.columns
		if len(cols) < 1 or not set(cols).issubset(set(x.columns)):
			raise ValueError("Todas las columnas deben existir en el dataframe.")
		if training_dataset:
			self.std_scaler = StandardScaler()
			self.std_scaler.fit(x[cols])
		if not self.std_scaler:
			raise ValueError(
				"El estandarizado de los datos debe ser entrenado antes de inferir, training_dataset = True"
			)
		x[cols] = self.std_scaler.transform(x[cols])
		return x
