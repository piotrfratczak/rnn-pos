import csv
import os

import nltk
import numpy as np
import tqdm


def load_data(data_root, dataset):
    if dataset == 'SpamSMS':
        return load_sms_ds(data_root)
    elif dataset == 'DisasterTweets':
        return load_tweets_ds(data_root)
    elif dataset in ['AmazonCells', 'IMDB', 'Yelp']:
        return load_sentiment(data_root, dataset)
    elif dataset == 'BBC_news':
        return load_bbc_news(data_root)


def load_sms_ds(data_root):
    texts, labels = [], []
    with open(os.path.join(data_root, 'sms_spam', 'SMSSpamCollection')) as f:
        for line in f:
            labels.append(line.split()[0].strip()), texts.append(' '.join(line.split()[1:]).strip())
    return texts, labels


def load_tweets_ds(data_root):
    texts, labels = [], []
    with open(os.path.join(data_root, 'nlp-disaster-tweets', 'train.csv')) as csv_file:
        for row in csv.DictReader(csv_file, delimiter=','):
            texts.append(row['text']), labels.append(int(row['target']))
    with open(os.path.join(data_root, 'nlp-disaster-tweets', 'test.csv')) as csv_file:
        test_data = {row['id']: row['text'] for row in csv.DictReader(csv_file, delimiter=',')}
    with open(os.path.join(data_root, 'nlp-disaster-tweets', 'sample_submission.csv')) as csv_file:
        test_labels = {row['id']: row['target'] for row in csv.DictReader(csv_file, delimiter=',')}
    for id in test_data.keys():
        texts.append(test_data[id]), labels.append(int(test_labels[id]))
    return texts, labels


def load_sentiment(data_root, ds):
    texts, labels = [], []
    file_names = {'AmazonCells': 'amazon_cells_labelled.txt', 'IMDB': 'imdb_labelled.txt', 'Yelp': 'yelp_labelled.txt'}
    with open(os.path.join(data_root, 'sentiment', file_names[ds])) as csv_file:
        for row in csv.reader(csv_file, delimiter='\t'):
            texts.append(row[0]), labels.append(int(row[1]))
    return texts, labels


def load_bbc_news(data_root):
    texts, labels = [], []
    with open(os.path.join(data_root, 'bbc', 'bbc-text.csv')) as csv_file:
        for row in csv.DictReader(csv_file, delimiter=','):
            texts.append(row['text']), labels.append(row['category'])
    return texts, labels


def load_glove_embedding_vec(data_root, dim):
    embedding_index = dict()
    with open(os.path.join(data_root, f"glove/glove.6B.{dim}d.txt")) as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            embedding_index[word] = np.asarray(values[1:], dtype='float32')
    return embedding_index


def download_necessary_nltk_packages():
    nltk.download(['averaged_perceptron_tagger', 'universal_tagset', 'tagsets'])
