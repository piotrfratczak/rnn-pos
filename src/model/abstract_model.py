from abc import ABC, abstractmethod

import torch
from torch import nn as nn


class AbstractSpamClassifier(nn.Module, ABC):

    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, drop_prob,
                 seq_len):
        super(AbstractSpamClassifier, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding_size = embedding_size
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(drop_prob)
        # dense layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # activation function
        self.sigmoid = nn.Sigmoid()

    @abstractmethod
    def forward(self, x, hidden):
        pass

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


class AbstractSpamClassifierWithTaggerIndexMapperAndDynamicGraph(AbstractSpamClassifier, ABC):
    def __init__(self, vocab_size, output_size, embedding_matrix, embedding_size,
                 hidden_dim, device, drop_prob, index_mapper, seq_len):
        super(AbstractSpamClassifierWithTaggerIndexMapperAndDynamicGraph, self).__init__(
            vocab_size, output_size, embedding_matrix, embedding_size,
            hidden_dim, device, drop_prob, seq_len)

        self.index_mapper = index_mapper
        self.tagger = None

    @property
    def lstm_mapping(self):
        raise NotImplementedError

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        indices_list = x.tolist()[0]
        list_of_words = self.index_mapper.indices_to_words(indices_list)
        list_of_tags = self.tagger.map_sentence(list_of_words)

        embeds = self.embedding(x)

        for i in range(self.seq_len):
            tag = list_of_tags[i]
            cell_input = embeds[:, i].view(batch_size, self.embedding_size)

            lstm_cell = self.lstm_mapping.get(tag, None)
            assert lstm_cell is not None

            hidden = lstm_cell(cell_input, hidden)

        lstm_out = hidden[0].contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -self.output_size:]
        return out, hidden
