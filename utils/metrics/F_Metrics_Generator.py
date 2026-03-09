import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	f1_score,
	precision_score,
	recall_score,
)


class Metrics_Generator:
	def __init__(self):
		self.results = Path(__file__).parent.parent.parent / "results"

	def generate(self, model_name, y_true, y_hat):
		if (self.results / model_name).exists():
			shutil.rmtree(self.results / model_name)
		(self.results / model_name).mkdir(parents=True, exist_ok=True)
		self.quantifiable_metrics(model_name, y_true, y_hat)
		self.confussion_matrix(model_name, y_true, y_hat)

	def quantifiable_metrics(self, model_name, y_true, y_hat):
		with open((self.results / model_name / "metrics.txt").resolve(), "w") as file:
			file.write(f"Accuracy: {accuracy_score(y_true, y_hat)}\n")
			file.write(f"Recall: {recall_score(y_true, y_hat, average='weighted')}\n")
			file.write(
				f"Precision: {precision_score(y_true, y_hat, average='weighted')}\n"
			)
			file.write(f"F1-Score: {f1_score(y_true, y_hat, average='weighted')}\n")
			file.write(f"Macro F1-Score: {f1_score(y_true, y_hat, average='macro')}\n")

	def confussion_matrix(self, model_name, y_true, y_hat):
		ConfusionMatrixDisplay.from_predictions(y_true, y_hat)
		plt.savefig((self.results / model_name / "confussion_matrix.png").resolve())
		plt.close()
