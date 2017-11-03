#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import os
import torch
import gc
from util.model import QaPredictModel
from util.nlp import QA, Tokenizer


DF_BATCH_SIZE = 5000
MODEL_BATCH_SIZE = 64


def get_model(X, emb):
    with open('model/weights.pickle', 'rb') as f:
        weights = pickle.load(f)

    return QaPredictModel({
        'num_word_features': len(X['paragraph_word_features'].iloc[0][0]),
        'num_features': len(X['paragraph_features'].iloc[0][0]),
        'hidden_size': 64,
        'doc_layers': 3,
        'question_layers': 3,
        'dropout_emb': 0.3,
        'dropout_rnn': 0.3,
        'tune_partial': 0,
        'cuda': False
    }, torch.Tensor(emb), weights)


def predict_df(df, vocab):
    print('Load df')
    qas = QA(tokenizer=Tokenizer(vocab=vocab))
    qas.load_df(df, answers=False)

    print('Export df')
    X, y = qas.get_Xy()
    model = get_model(X, qas.tokenizer.vocab.embeddings)
    del qas
    gc.collect()

    print('Predict df')
    return [p.answer.text for p in model.batch_size(MODEL_BATCH_SIZE).predicts(X, y)]

if __name__ == '__main__':
    DATA_FILE = os.environ.get('INPUT', 'data/train.csv')
    PREDICTION_FILE = os.environ.get('OUTPUT', 'data/result.csv')

    df = pd.DataFrame.from_csv(DATA_FILE, sep=',', index_col=None, encoding='utf8')
    df = df[['paragraph_id', 'question_id', 'paragraph', 'question']]

    with open('model/vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)

    predicts = []
    batches = df.index.groupby(np.arange(len(df)) // DF_BATCH_SIZE)
    for i, idx in batches.items():
        print('[{}/{}] Process batch'.format(i + 1, len(batches)))
        df_batch = df.loc[idx]
        predicts += predict_df(df_batch, vocab)

    df['answer'] = predicts
    df.set_index(['paragraph_id', 'question_id'])['answer'].to_csv(PREDICTION_FILE, header=True, encoding='utf8')
