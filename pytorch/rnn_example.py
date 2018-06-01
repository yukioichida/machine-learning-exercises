import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules.lstm_tagger import LSTMTagger


def prepare_sequence(sequence, to_index):
    index = [to_index[w] for w in sequence]
    return torch.tensor(index, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

# Creating a vocabulary
word_to_index = {}

for sentence, _ in training_data:
    for word in sentence:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

# word_to_index = {'The': 0, 'dog': 1, 'ate': 2, 'the': 3, 'apple': 4, 'Everybody': 5, 'read': 6, 'that': 7, 'book': 8}

label_to_index = {"DET": 0, "NN": 1, "V": 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_index), len(label_to_index))
# negative log likelihood loss
loss_function = nn.NLLLoss()

# stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.1)

# No train, just for see tensor formats
with torch.no_grad():
    # retrieve index from each word contained in first sentence
    inputs = prepare_sequence(training_data[0][0], word_to_index)
    tag_scores = model(inputs)
    print("Inputs: "+str(inputs))
    print("Tag scores predicted: "+str(tag_scores))

print("Training ...")
for epoch in range(300):
    for sentence, label in training_data:
        model.zero_grad()

        model.hidden = model.init_hidden(HIDDEN_DIM)

        x = prepare_sequence(sentence, word_to_index)
        y = prepare_sequence(label, label_to_index)

        y_predicted = model(x)

        loss = loss_function(y_predicted, y)
        print(loss.item())

        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_index)
    scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(scores)

