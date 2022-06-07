import torch

class Runner:
	def __init__(self, file_path):
		self.file_path = file_path
		self.data = self.read_data_in_format()
		self.vocab = list(set(self.data))
		print(self.vocab)

	def read_data_in_format(self):
		with open(self.file_path, "r") as f:
			return f.read()
