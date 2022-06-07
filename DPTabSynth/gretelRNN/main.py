import torch
import pandas as pd
import numpy as np
from .model import GeneratorModel
from tqdm import tqdm
import os

class TabDataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x, self.y = x, y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


class Runner:
	def __init__(self, file_path, batch_size=16, input_dim=20, shuffle=True, embedding_dim=10, lr=0.001):
		self.input_dim = input_dim
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.file_path = file_path
		self.continuous_present = False
		self.data = self.read_data_in_format()
		self.vocab = list(set(self.data))
		if self.continuous_present:
			self.vocab = list(set(self.vocab + list(range(10))))
		self.char_present = bool([1 for i in self.vocab if not str(i).isdigit()])
		if self.char_present:
			self.vocab = [str(i) for i in self.vocab]
		self.vocab = sorted(self.vocab)
		self.chr2idx = {i:j for j,i in enumerate(self.vocab)}
		self.idx2chr = {j:i for j,i in enumerate(self.vocab)}
		self.model = self.get_model()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)#torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
		self.x, self.y = self.create_batched_dataset()
		self.criterion = torch.nn.CrossEntropyLoss()

	def get_model(self):
		return GeneratorModel(len(self.vocab), 24)

	def read_data_in_format(self):
		df = pd.read_csv(self.file_path)
		df = df.values
		for col in df.T:
			try:
				col = col.astype(np.float32)
				self.continuous_present = True
			except:
				pass
		with open(self.file_path, "r") as f:
			dat = "\n".join([i.strip() for i in f.readlines()][1:])
			return dat

	def create_batched_dataset(self):
		tokens = [self.chr2idx[i] for i in self.data]
		x_y = [[tokens[i:i+self.input_dim], tokens[i+self.input_dim+1]] for i in range(0, len(tokens)-self.input_dim-1)]
		x = np.array([i[0] for i in x_y])
		y = np.array([i[1] for i in x_y])
		return x, y

	def fit(self, num_samples, epochs, prev_state_return=True, continue_train=False):
		self.model.train()
		self.x = torch.from_numpy(self.x[:num_samples])
		self.y = torch.from_numpy(self.y[:num_samples])
		self.y = self.y.long()
		self.x, self.y = self.x.to(self.device), self.y.to(self.device)
		dataset = TabDataset(self.x, self.y)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
		if os.path.exists("best_model.pt") and prev_state_return:
			return self.load_best_state()
		if prev_state_return==False and continue_train:
			self.load_best_state()
		print("-"*10+"fitting"+"-"*10)
		prev_loss = []
		best_loss = 1e5
		for epoch in range(epochs):
			running_loss = 0.0
			running_acc = 0.01
			bar = tqdm(dataloader, total=len(dataloader))
			for i, (batch_x, batch_y) in enumerate(bar):
				self.optimizer.zero_grad()
				out = self.model(batch_x)
				loss = self.criterion(out, batch_y)
				loss.backward()
				self.optimizer.step()
				running_loss += loss.item()
				pred = torch.argmax(out, dim = 1)
				acc = torch.mean((pred == batch_y).float())
				running_acc += acc.item()
				bar.set_description(str({"epoch":epoch,"loss":round(running_loss/(i+1),3),"acc":round(running_acc/(i+1),3)}))
			bar.close()
			running_loss = running_loss/(i+1)
			if running_loss<best_loss:
				best_loss = running_loss
				torch.save(self.model, "best_model.pt")
			if len(prev_loss)<3:
				prev_loss.append(running_loss)
			else:
				if running_loss>=prev_loss[0] and running_loss>=prev_loss[1] and running_loss>=prev_loss[2]:
					print("Stopped Training based on early stopping...")
					break
				else:
					prev_loss.append(running_loss)
					prev_loss = prev_loss[1:]
		return self.model

	def load_best_state(self):
		self.model = torch.load("best_model.pt")
		return self.model

	def generate_synthetics(self, rows=1000, input_x=None, sep='\n'):
		rows = rows+1
		if input_x is None:
			input_x = self.x[np.random.choice(np.arange(len(self.x)))]
		input_x = input_x.tolist()
		self.model.eval()
		data = []
		sep_count = 0
		bar = tqdm(desc="Generating",total=rows-1)
		count_from_prev_sep = 0
		while(True):
			x = torch.from_numpy(np.array([input_x])).to(self.device)
			out = self.model(x)
			next_chr = torch.argmax(out, dim = 1).item()
			data.append(next_chr)
			input_x.append(next_chr)
			input_x = input_x[1:]
			if next_chr==self.vocab.index(sep):
				count_from_prev_sep = 0
				sep_count +=1
				if sep_count!=1:
					bar.update(1)
			else:
				count_from_prev_sep +=1
			if count_from_prev_sep==1000:
				print("Error can't come to next row exit")
				exit()
			if sep_count==rows:
				break
		bar.close()
		return "".join([self.idx2chr[i] for i in data[data.index(self.vocab.index(sep)):]])

if __name__ == '__main__':
	main()