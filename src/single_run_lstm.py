import os
import torch
import logging
import pathlib
import torch.nn as nn

from src.test import test
from src.train import train, save_model
from src.data_loading import load_data
from src.utils import Selector, get_model, get_embedding_vectors, prepare_dataloaders, \
    add_parameters_to_test_results


def single_run_lstm(params, embeddings):
    data_root = os.path.join(pathlib.Path(__file__).parent.parent, 'data')
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    logging.info(
        f"Using parameters: sequence_length={params['sequence_length']} embedding_size={params['embedding_size']} "
        f"epochs={params['epochs']} learning_rate={params['learning_rate']} padding={params['padding']}")
    X, y = load_data(data_root, params['dataset'])
    train_loader, val_loader, test_loader, tokenizer, output_size = prepare_dataloaders(X, y, params)
    embedding_matrix = get_embedding_vectors(tokenizer, params['embedding_size'], embeddings)
    vocab_size = len(tokenizer.word_index) + 1

    run_results = []
    for model_idx in range(len(Selector)):
        model_name = Selector(model_idx).name
        logging.info(f"Model name: {model_name}")

        model = get_model(Selector(model_idx), vocab_size, output_size, embedding_matrix, params['embedding_size'],
                          params['hidden_dim'], device, params['drop_prob'], tokenizer, params['sequence_length'])
        criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        model, train_stats = train(model, params['epochs'], train_loader, val_loader, device, optimizer, criterion)
        save_model(model, f'weights/{model_name}_{params["dataset"]}_seq:{params["sequence_length"]}'
                          f'_emb:{params["embedding_size"]}_lr:{params["learning_rate"]}_pad:{params["padding"]}')
        test_results = test(model, test_loader, device, criterion)

        test_results = add_parameters_to_test_results(
            test_results, model_name, params['sequence_length'], params['embedding_size'],
            train_stats['epoch_min_loss'], params['learning_rate'], params['padding'], params['dataset']
        )

        run_results.append(test_results)

    return run_results
