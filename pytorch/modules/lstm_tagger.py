import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, labels_size):
        super(LSTMTagger, self).__init__()
        # Layer 1
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Layer 2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # Output Layer
        self.hiddenToOutput = nn.Linear(hidden_dim, labels_size)
        # Initialize hidden state of lstm
        self.hidden = self.init_hidden(hidden_dim)

    def init_hidden(self, hidden_dim):
        return (torch.zeros(1, 1, hidden_dim),
                torch.zeros(1, 1, hidden_dim))

    def forward(self, sentence):
        embeddings = self.word_embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeddings.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hiddenToOutput(lstm_out.view(len(sentence), -1))
        scores = F.log_softmax(tag_space, 1)
        return scores

