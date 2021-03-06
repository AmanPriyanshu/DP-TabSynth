import torch
#criterion = torch.nn.CrossEntropyLoss()
class GeneratorModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_dim):
        super(GeneratorModel, self).__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.rnn = torch.nn.LSTM(embedding_dim, 2*embedding_dim, 2, batch_first=True)
        n = 2*embedding_dim*input_dim
        self.linearLayer1 = torch.nn.Linear(n, embedding_dim)
        self.linearLayer2 = torch.nn.Linear(embedding_dim, vocab_size)
        self.relu = torch.nn.ReLU()

    def forward(self, og_x):
        x = self.embedding_layer(og_x)
        x = self.dropout(x)
        x, hidden = self.rnn(x)
        x = self.dropout(x)
        x = torch.reshape(x, (len(og_x), -1))
        x = self.linearLayer1(x)
        x = self.relu(x)
        x = self.linearLayer2(x)
        return x

if __name__ == '__main__':
    x =torch.tensor([[0, 1, 2, 3, 4, 0, 1]])
    model = GeneratorModel(5, 10)
    print(model(x).shape)
