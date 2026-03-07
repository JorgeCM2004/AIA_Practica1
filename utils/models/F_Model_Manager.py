from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

POSSIBLE_MODELS = Literal["Random Forest", "XGBoost", "KNN", "Logistic Regression"]


class Model_Manager:
	def __init__(
		self, models: list[POSSIBLE_MODELS] | Literal["All"] = "All", seed=None
	):
		self.models = []
		if models == "All":
			models = list(POSSIBLE_MODELS)
		self.models_names = [model for model in models if model in set(POSSIBLE_MODELS)]
		for model in self.models_names:
			match model:
				case "Random Forest":
					self.models.append(RandomForestClassifier(random_state=seed))
				case "XGBoost":
					self.models.append(XGBClassifier(random_state=seed))
				case "KNN":
					self.models.append(KNeighborsClassifier())
				case "Logistic Regression":
					self.models.append(LogisticRegression(random_state=seed))

	def train(self, x: pd.DataFrame, y: pd.Series):
		for model in self.models:
			model.fit(x, y)

	def predict(self, x):
		y_hats = []
		for model in self.models:
			y_hats.append(model.predict(x))
		return self.models_names, y_hats
