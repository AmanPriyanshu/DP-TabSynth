import pandas as pd
import numpy as np

data = {}
df = pd.read_csv("mushroom.csv", sep=";")
df = df.dropna(axis = 'columns')
features = list(df.columns)
df = df.values

for idx, col in enumerate(features):
	if type(df.T[idx][0])==str:
		if(df.T[idx][0][0]=='['):
			list_of_things = [[j for j in i[1:-1].split(',')] for i in df.T[idx]]
			if idx in [3,8,9]:
				data[col+'-x'] = [i[0] for i in list_of_things]
				data[col+'-y'] = [i[1] if len(i)==2 else None for i in list_of_things]
			else:
				if not list_of_things[0][0].isdigit():
					vocab = []
					for arr in list_of_things:
						vocab.extend(arr)
						vocab = list(set(vocab))
					one_hot_encode = np.zeros((len(df), len(vocab)))
					for i in range(len(df)):
						val = list_of_things[i]
						for k in val:
							one_hot_encode[i][vocab.index(k)] = 1
					for j, item in enumerate(vocab):
						data[col+'-'+item.strip()] = one_hot_encode.T[j]
		else:
			data[col] = [i for i in df.T[idx]]
	else:
		data[col] = [i for i in df.T[idx]]

data = pd.DataFrame(data)
data.to_csv("mushroo.csv", index=False)