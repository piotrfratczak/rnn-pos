import logging
import pickle

import numpy as np
import torch.nn as nn
from tqdm import tqdm

from src.utils import exec_batch_roberta_model, exec_batch_lstm_models


def save_model(model):
    if repr(model) == "Roberta":
        model.save_weights()
    else:
        with open('weights/model.pickle', 'wb') as f:
            pickle.dump(model, f)


def train(model, epochs, train_loader, val_loader, device, optimizer, criterion, clip=None):
    model.to(device)
    epoch_min_loss, min_loss_val_ds, loaders = None, np.inf, {'train': train_loader, 'val': val_loader}
    for epoch in tqdm(range(1, epochs + 1)):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            losses = []
            for inputs, labels in loaders[phase]:
                if repr(model) == "Roberta":
                    output = exec_batch_roberta_model(inputs, device, model)
                else:
                    output = exec_batch_lstm_models(inputs, device, model)

                loss = criterion(output, labels.to(device).float())
                if phase == 'train':
                    loss.backward()
                    if clip:
                        nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                losses.append(loss.item())

            avg_loss_phase = np.mean(losses)
            if phase == 'val' and avg_loss_phase < min_loss_val_ds:
                min_loss_val_ds, epoch_min_loss = avg_loss_phase, epoch
                save_model(model)

            logging.info(f"epoch = {epoch}, loss for phase {phase} = {round(avg_loss_phase, 2)}")

    return {'epoch_min_loss': epoch_min_loss}
