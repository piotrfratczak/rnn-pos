import logging
import pickle

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.utils import get_one_hot_label, exec_batch_roberta_model, exec_batch_lstm_models


def test(model, test_loader, device, criterion):
    if repr(model) == "Roberta":
        model.load_weights()
    else:
        with open('weights/model.pickle', 'rb') as f:
            model = pickle.load(f)

    model.eval()
    test_lab_vec, test_pred_vec, test_losses = [], [], []

    for inputs, labels in test_loader:
        if repr(model) == "Roberta":
            output = exec_batch_roberta_model(inputs, device, model)
        else:
            output = exec_batch_lstm_models(inputs, device, model)

        loss = criterion(output, labels.to(device).float())
        test_losses.append(loss.item())

        predictions = list(map(get_one_hot_label, output.tolist()))
        test_pred_vec.extend(predictions), test_lab_vec.extend(labels.tolist())

    accuracy = accuracy_score(y_true=test_lab_vec, y_pred=test_pred_vec)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true=test_lab_vec, y_pred=test_pred_vec, beta=1,
                                                       average='weighted')

    test_loss = np.mean(test_losses)

    logging.info(
        f'\nAccuracy: {accuracy:.4f}'
        f'\nAverage precision score: {prec:.4f}'
        f'\nAverage recall score: {rec:0.4f}'
        f'\nAverage f1-recall score: {f1:0.4f}'
        f'\nTest loss: {test_loss:.2f}'
    )

    return {
        "accuracy": accuracy,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "test_loss": test_loss
    }
