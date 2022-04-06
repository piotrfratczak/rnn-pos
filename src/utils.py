import json
import logging
from datetime import datetime
from enum import Enum
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset

from src.data_loading import load_glove_embedding_vec
from src.encoder import Encoder
from src.model.model import SpamClassifierLstmLayer, SpamClassifierSingleLstmCell, SpamClassifierLstmPosUniversal, \
    SpamClassifierLstmPosPenn


class Selector(Enum):
    LSTM_Layer = 0
    LSTM_Single_Cell = 1
    LSTM_POS_Penn = 2
    LSTM_POS_Universal = 3


class IndexMapper:
    def __init__(self, tokenizer):
        self.dictionary = json.loads(tokenizer.get_config()['index_word'])

    def index_to_word(self, index):
        return "-" if index == 0 else self.dictionary[str(index)]

    def indices_to_words(self, indices):
        list_of_words = [None] * len(indices)
        for i in range(len(indices)):
            list_of_words[i] = self.index_to_word(indices[i])
        return list_of_words


class CustomRobertaDataset(Dataset):
    def __init__(self, corpus: List[str], labels: np.ndarray):
        super().__init__()
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        length = len(self.labels)
        return length

    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = self.labels[idx]
        return text, label


def get_model(selector, vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device, drop_prob,
              tokenizer, seq_len, n_layers=2):
    if selector == Selector.LSTM_Layer:
        return SpamClassifierLstmLayer(vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device,
                                       drop_prob, n_layers, seq_len)
    elif selector == Selector.LSTM_Single_Cell:
        return SpamClassifierSingleLstmCell(vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim,
                                            device, drop_prob, seq_len)
    elif selector == Selector.LSTM_POS_Penn:
        return SpamClassifierLstmPosPenn(vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim, device,
                                         drop_prob, IndexMapper(tokenizer), seq_len)

    elif selector == Selector.LSTM_POS_Universal:
        return SpamClassifierLstmPosUniversal(vocab_size, output_size, embedding_matrix, embedding_size, hidden_dim,
                                              device, drop_prob, IndexMapper(tokenizer), seq_len)
    else:
        raise ValueError(f"Unknown selector value: {selector}")


def get_embedding_vectors(input_tokenizer, dim, embeddings):
    embedding_index = embeddings[dim]

    word_index = input_tokenizer.word_index
    new_embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            new_embedding_matrix[i] = embedding_vector

    return new_embedding_matrix


def get_embeddings():
    return {
        50: load_glove_embedding_vec("data/", 50),
        100: load_glove_embedding_vec("data/", 100),
        300: load_glove_embedding_vec("data/", 300)
    }


def text_preprocessing(params, X, y):
    encoder = Encoder()
    y_one_hot = encoder.fit_transform(y)
    output_size = len(y_one_hot[0])

    if params['mode'] == 'debug':
        X, y_one_hot = X[:params['debug_ds_len']], y_one_hot[:params['debug_ds_len']]

    tokenizer = Tokenizer(lower=params['tokenizer_lower'])
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = [row[:params['sequence_length']] for row in X]

    X, y_one_hot = np.array(X, dtype=object), np.array(y_one_hot, dtype=object)
    X = pad_sequences(X, maxlen=params['sequence_length'], padding=params['padding'])

    y_one_hot = np.asarray(y_one_hot, dtype=np.float32)
    return X, y_one_hot, tokenizer, output_size


def roberta_text_preprocessing(params, X, y):
    encoder = Encoder()
    y_one_hot = encoder.fit_transform(y)
    output_size = len(y_one_hot[0])

    if params['mode'] == 'debug':
        X, y_one_hot = X[:params['debug_ds_len']], y_one_hot[:params['debug_ds_len']]

    # 512 is maximum possible length for Roberta
    if params['sequence_length'] > 512:
        raise ValueError("Sequence length for Roberta must not be bigger than 512")

    X = [row[:params['sequence_length']] for row in X]
    y_one_hot = np.asarray(y_one_hot, dtype=np.float32)
    return X, y_one_hot, output_size


def prepare_data_loaders_and_tokenizer(X, y, params, for_roberta=False):
    if not for_roberta:
        X, y_one_hot, tokenizer, output_size = text_preprocessing(params, X, y)
    else:
        X, y_one_hot, output_size = roberta_text_preprocessing(params, X, y)
        tokenizer = None

    x_train, x_test, y_train, y_test = train_test_split(X, y_one_hot, train_size=params['train_ratio'], random_state=7)

    split_idx = int(params['val_ratio'] / (params['val_ratio'] + params['test_ratio']) * len(x_test))
    x_val, x_test, y_val, y_test = x_test[:split_idx], x_test[split_idx:], y_test[:split_idx], y_test[split_idx:]

    if not for_roberta:
        train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        val_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
        test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    else:
        train_data = CustomRobertaDataset(x_train, y_train)
        val_data = CustomRobertaDataset(x_val, y_val)
        test_data = CustomRobertaDataset(x_test, y_test)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=params['batch_size'])
    val_loader = DataLoader(val_data, shuffle=True, batch_size=params['batch_size'])
    test_loader = DataLoader(test_data, shuffle=True, batch_size=params['batch_size'])

    return train_loader, val_loader, test_loader, tokenizer, output_size


def get_one_hot_label(nn_output: List[float]):
    one_hot = [1 if x == max(nn_output) else 0 for x in nn_output]
    return one_hot


def add_parameters_to_test_results(test_results, model_name, sequence_length,
                                   embedding_size, epochs, learning_rate, padding, dataset):
    test_results["model"] = model_name
    test_results["sequence_length"] = sequence_length
    test_results["embedding_size"] = embedding_size
    test_results["epochs"] = epochs
    test_results["learning_rate"] = learning_rate
    test_results["padding"] = padding
    test_results["dataset"] = dataset

    return test_results


def save_results_to_csv(results: List[Dict], filename):
    current_date = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    filename = f"results/{filename}_{current_date}.csv"

    logging.info(f"Saving results in file: {filename}")
    df = pd.DataFrame(data=results)
    df.to_csv(filename, index=False)


def exec_batch_roberta_model(inputs, device, model):
    model.zero_grad()

    features = [model.get_roberta_predictions(model.encode(sample)).to(device)[:, 0, :] for sample in inputs]
    features = torch.stack(features)
    output = model(features)
    output = torch.squeeze(output)
    return output


def exec_batch_lstm_models(inputs, device, model):
    hidden = model.init_hidden(len(inputs))
    hidden = tuple([e.data for e in hidden])
    inputs = inputs.to(device)
    model.zero_grad()
    output, hidden = model(inputs, hidden)
    return output
