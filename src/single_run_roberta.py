import logging
import os
import pathlib

import torch
from torch import nn

from src.data_loading import load_data
from src.model.roberta import RobertaClassifier
from src.test import test
from src.train import train
from src.utils import prepare_data_loaders_and_tokenizer, add_parameters_to_test_results


def single_run_roberta(params):
    data_root = os.path.join(pathlib.Path(__file__).parent.parent, 'data')
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    logging.info(
        f"Using parameters: epochs={params['epochs']},  learning_rate={params['learning_rate']}, "
        f"hidden_size={params['hidden_dim']}, sequence_length={params['sequence_length']}")
    X, y = load_data(data_root, params['dataset'])
    train_loader, val_loader, test_loader, _, output_size = prepare_data_loaders_and_tokenizer(X, y, params,
                                                                                               for_roberta=True)

    model = RobertaClassifier(output_size, params['hidden_dim'])
    criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    train_stats = train(model, params['epochs'], train_loader, val_loader, device, optimizer, criterion)
    test_results = test(model, test_loader, device, criterion)

    test_results = add_parameters_to_test_results(
        test_results, 'Roberta Classifier', params['sequence_length'], None, train_stats['epoch_min_loss'],
        params['learning_rate'], None, params['dataset']
    )

    return [test_results]
