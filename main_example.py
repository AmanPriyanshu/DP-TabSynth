from DPTabSynth.gretelRNN.main import Runner

if __name__ == '__main__':
	file_path = "./data/bank.csv"
	runner = Runner(file_path, lr=0.01, input_dim=40, cat_based=True)#, lr=0.05)
	model = runner.fit(-1, 100)
	data = runner.generate_synthetics()
	print(data)