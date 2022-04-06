import pickle

import torch
from torch import nn


class RobertaClassifier(nn.Module):
    class RobertaWrapper:
        def __init__(self):
            self.model = torch.hub.load('pytorch/fairseq', 'roberta.base')

        def get_prediction(self, x):
            with torch.no_grad():
                output = self.model.extract_features(x)
            return output

        def encode(self, x):
            return self.model.encode(x)

    def __init__(self, num_classes, hidden_size=128):
        super().__init__()
        self.roberta = RobertaClassifier.RobertaWrapper()

        self.fc1 = nn.Linear(768, hidden_size)  # roberta outputs 768 features from the last layer
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def get_roberta_predictions(self, x):
        return self.roberta.get_prediction(x)

    def encode(self, x):
        return self.roberta.encode(x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

    # This method exists to ensure identical interfaces with other models
    def init_hidden(self, batch_size):
        pass

    def save_weights(self):
        with open('weights/model.pickle', 'wb') as f:
            pickle.dump(self.roberta.model.model, f)

    def load_weights(self):
        with open('weights/model.pickle', 'rb') as f:
            self.roberta.model.model = pickle.load(f)

    def __repr__(self):
        return "Roberta"
