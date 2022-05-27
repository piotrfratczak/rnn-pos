import json
import time
import logging
from itertools import product

from src.train.single_run_lstm import single_run_lstm
from src.utils.data_loading import download_nltk_packages
from src.train.single_run_roberta import single_run_roberta
from src.utils.utils import save_results_to_csv, get_embeddings


logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.INFO)


if __name__ == "__main__":
    start = time.perf_counter()

    with open('configs/lstm_config.json', 'r') as lstm_fp, open('configs/roberta_config.json', 'r') as roberta_fp:
        all_possible_params_lstm = json.load(lstm_fp)
        all_possible_params_roberta = json.load(roberta_fp)

    # LSTM
    if all_possible_params_lstm['run'] in ['y', 'Y', '1']:
        download_nltk_packages()
        results = []
        embeddings = get_embeddings()
        for idx, single_params in enumerate(product(*all_possible_params_lstm.values()), 1):
            logging.info(f"Progress for lstm = {idx}/{len(list(product(*all_possible_params_lstm.values())))}")
            single_params_dict = dict(zip(all_possible_params_lstm, single_params))
            run_results = single_run_lstm(single_params_dict, embeddings)
            results.extend(run_results)
            if single_params_dict['mode'] != 'debug':
                save_results_to_csv(results, f'lstm_{idx}')

        save_results_to_csv(results, 'lstm')

    # Roberta
    if all_possible_params_roberta['run'] in ['y', 'Y', '1']:
        results = []
        for idx, single_params in enumerate(product(*all_possible_params_roberta.values()), 1):
            logging.info(f"Progress for roberta {idx}/{len(list(product(*all_possible_params_roberta.values())))}")
            single_params_dict = dict(zip(all_possible_params_roberta, single_params))
            run_results = single_run_roberta(single_params_dict)
            results.extend(run_results)
            if single_params_dict['mode'] != 'debug':
                save_results_to_csv(results, f'roberta_{idx}')

        save_results_to_csv(results, 'roberta')

    end = time.perf_counter()
    print(f'Exec time: {end - start}')
