import os
import torch
import numpy as np
import pandas as pd
from drqa.model import DocReaderModel
from util.squad import f1_score, f1_score_mean
from util.nlp import QA, Tokenizer
from tqdm import *


class QaPredictLiveModel:
    def __init__(self, opt, vocab, weights):
        self.qas = QA(tokenizer=Tokenizer(vocab=vocab))

        X, y, emb = self.__get_Xy_emb('init', 'init')
        model_opt = opt.copy()
        model_opt.update({
            'num_word_features': len(X['paragraph_word_features'].iloc[0][0]),
            'num_features': len(X['paragraph_features'].iloc[0][0]),
            'tune_partial': 0
        })
        self.model = QaPredictModel(model_opt, torch.Tensor(emb), weights)

    def predict(self, paragraph, question):
        X, y, emb = self.__get_Xy_emb(paragraph, question)
        self.model.model.network.embedding.weight.data = torch.cat([
            self.model.model.network.embedding.weight.data,
            torch.Tensor(emb[len(self.model.model.network.embedding.weight.data):])
        ])

        return list(self.model.predicts(X, y, tqdm_show=False))[0]

    def __get_Xy_emb(self, paragraph, question):
        self.qas.paragraph_questions = []
        p = self.qas.add_paragraph(paragraph)
        q = self.qas.add_question(question)
        self.qas.add(p, q)

        X, y = self.qas.get_Xy(tqdm_show=False)
        emb = self.qas.tokenizer.vocab.embeddings

        return X, y, emb


class QaAnswer:
    def __init__(self, text, score_start, score_end):
        self.text = text
        self.score = score_start * score_end
        self.score_start = score_start
        self.score_end = score_end

    def __repr__(self):
        return self.text.__repr__()


class QaAnswerGroup:
    def __init__(self, answer):
        self.answers = [answer]
        self.scores_start = [answer.score_start]
        self.scores_end = [answer.score_end]

        self.score = answer.score
        self.score_start = answer.score_start
        self.score_end = answer.score_end

    def match_answer(self, answer, f1_threshold=0.5):
        if answer in self.answers:
            return True

        if f1_score(self.answers[0].text, answer.text) >= f1_threshold:
            self.answers.append(answer)

            if answer.score_start not in self.scores_start:
                self.scores_start.append(answer.score_start)
                self.score_start += answer.score_start

            if answer.score_end not in self.scores_end:
                self.scores_end.append(answer.score_end)
                self.score_end += answer.score_end

            self.score = self.score_start * self.score_end * len(self.answers)

            return True

        return False

    def __repr__(self):
        return (self.answers[0], self.score).__repr__()


class QaPredict:
    def __init__(self, scores_start, scores_end, paragraph, paragraph_spans, max_len=30):
        scores = torch.ger(scores_start, scores_end)
        scores.triu_().tril_(max_len - 1)
        self.scores = scores.numpy()
        self.paragraph = paragraph
        self.paragraph_spans = paragraph_spans
        self.scores_start = scores_start.numpy()[:len(paragraph_spans)]
        self.scores_end = scores_end.numpy()[:len(paragraph_spans)]
        self.answer = self.__get_answer_by_idx(np.argmax(self.scores))

    def get_scores(self):
        tokens = [self.paragraph[token_start:token_end] for token_start, token_end in self.paragraph_spans]
        return pd.DataFrame({'start': self.scores_start, 'end': self.scores_end}, index=tokens)

    def plot_scores(self):
        scores = self.get_scores()
        return scores.iloc[::-1].plot.barh(figsize=(10, 0.2 * len(scores.index)), width=0.8)

    def __get_answer_by_idx(self, idx):
        s_idx, e_idx = np.unravel_index(idx, self.scores.shape)
        s_offset, e_offset = self.paragraph_spans[s_idx][0], self.paragraph_spans[e_idx][1]

        return QaAnswer(self.paragraph[s_offset:e_offset], self.scores_start[s_idx], self.scores_end[e_idx])

    def plot_top_answers(self, top_n=10):
        top_answers = self.get_top_answers(top_n)
        df = pd.Series([a.score for a in top_answers], index=[a.text for a in top_answers])

        return df.iloc[::-1].plot.barh(figsize=(10, 0.3 * len(df)), width=0.8)

    def get_top_answers(self, top_n=10):
        return [self.__get_answer_by_idx(idx) for idx in (-self.scores).argsort(axis=None)[:top_n]]

    def get_top_answer_groups(self, top_n=10, f1_threshold=0.1):
        groups = []

        for a in self.get_top_answers(top_n=top_n):
            matched = False
            for g in groups:
                if g.match_answer(a, f1_threshold=f1_threshold):
                    matched = True

            if not matched:
                groups.append(QaAnswerGroup(a))

        return sorted(groups, key=lambda g: -g.score)

    def __repr__(self):
        return self.answer.__repr__()


class QaPredictModel:
    OPT = {
        'cuda': True,
        'batch_size': 64,
        'grad_clipping': 10,
        'weight_decay': 0,
        'question_merge': 'self_attn',
        'doc_layers': 3,
        'question_layers': 3,
        'hidden_size': 64,
        'use_qemb': True,
        'concat_rnn_layers': True,
        'dropout_emb': 0.3,
        'dropout_rnn': 0.3,
        'dropout_rnn_output': True,
        'max_len': 30,
        'rnn_type': 'lstm'
    }

    def __init__(self, opt, embedding, weights=None):
        model_opt = QaPredictModel.OPT.copy()
        model_opt.update(opt)
        model_opt.update({
            'embedding_dim': embedding.size(1)
        })

        self.model = DocReaderModel(model_opt, embedding)
        if weights:
            self.model.load_weights(weights)

        if self.model.opt['cuda']:
            self.model.cuda()

    def batch_gen(self, X, y, train=True):
        return QaBatchGen(X, y, batch_size=self.model.opt['batch_size'], cuda=self.model.opt['cuda'], train=train)

    def predicts(self, X, y, tqdm_show=True):
        batches = self.batch_gen(X, y, train=False)
        if tqdm_show:
            batches = tqdm(batches, total=len(batches), unit='batch')

        for batch in batches:
            paragraphs = batch[-2]
            paragraphs_spans = batch[-1]
            score_start, score_end = self.model.predict(batch)

            for i in range(score_start.size(0)):
                yield QaPredict(
                    score_start[i],
                    score_end[i],
                    paragraphs[i],
                    paragraphs_spans[i],
                    max_len=self.model.opt['max_len']
                )

    def train(self, X, y):
        batches = self.batch_gen(X, y)
        for batch in tqdm(batches, total=len(batches), unit='batch'):
            self.model.update(batch)

    def validate(self, X, y):
        predicts = [p.answer.text for p in self.predicts(X, y)]
        return f1_score_mean(predicts, y['answer']), predicts

    def train_and_validate(self, X_train, y_train, X_text, y_test):
        print('Train')
        self.train(X_train, y_train)

        print('Validate')
        return self.validate(X_text, y_test)

    def epoch(self, model_name, X_train, y_train, X_test, y_test):
        score, predicts = self.train_and_validate(X_train, y_train, X_test, y_test)
        print('Train Loss: ' + str(self.get_train_loss()))
        print('Score: ' + str(score))
        print('')
        self.save(os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'models', model_name + '-' + str(score) + '.pt')
        ))

        return score, predicts

    def get_network(self):
        return self.model.network

    def get_train_loss(self):
        return self.model.train_loss.avg

    def save(self, filename):
        self.model.save_weights(filename)

    def batch_size(self, batch_size):
        self.model.opt['batch_size'] = batch_size
        return self

    def max_len(self, max_len):
        self.model.opt['max_len'] = max_len
        return self


class QaBatchGen:
    def __init__(self, X, y, batch_size=32, cuda=True, train=True):
        self.batch_size = batch_size
        self.cuda = cuda
        self.train = train

        if train:
            # shuffle
            X = X.sample(frac=1)
            y = y.loc[X.index]

        self.X = X
        self.y = y
        self.batches = X.index.groupby(np.arange(len(X)) // self.batch_size)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for _, idx in self.batches.items():
            batch_size = len(idx)
            X = self.X.loc[idx]
            y = self.y.loc[idx]

            paragraph_len = int(X['paragraph_embeddings'].map(len).max())
            paragraph = torch.LongTensor(batch_size, paragraph_len).fill_(0)
            for i, p in enumerate(X['paragraph_embeddings']):
                paragraph[i, :len(p)] = torch.LongTensor(p)

            paragraph_word_features_len = len(X['paragraph_word_features'].iloc[0][0])
            paragraph_word_features = torch.Tensor(batch_size, paragraph_len, paragraph_word_features_len).fill_(0)
            for i, features in enumerate(X['paragraph_word_features']):
                for j, f in enumerate(features):
                    paragraph_word_features[i, j, :] = torch.Tensor(f)

            paragraph_features_len = len(X['paragraph_features'].iloc[0][0])
            paragraph_features = torch.Tensor(batch_size, paragraph_len, paragraph_features_len).fill_(0)
            for i, features in enumerate(X['paragraph_features']):
                for j, f in enumerate(features):
                    paragraph_features[i, j, :] = torch.Tensor(f)

            question_len = int(X['question_embeddings'].map(len).max())
            question = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, q in enumerate(X['question_embeddings']):
                question[i, :len(q)] = torch.LongTensor(q)

            question_word_features_len = len(X['question_word_features'].iloc[0][0])
            question_word_features = torch.Tensor(batch_size, question_len, question_word_features_len).fill_(0)
            for i, features in enumerate(X['question_word_features']):
                for j, f in enumerate(features):
                    question_word_features[i, j, :] = torch.Tensor(f)

            paragraph_mask = torch.eq(paragraph, 0)
            question_mask = torch.eq(question, 0)

            if self.cuda:
                paragraph = paragraph.pin_memory()
                paragraph_word_features = paragraph_word_features.pin_memory()
                paragraph_features = paragraph_features.pin_memory()
                paragraph_mask = paragraph_mask.pin_memory()
                question = question.pin_memory()
                question_word_features = question_word_features.pin_memory()
                question_mask = question_mask.pin_memory()

            paragraph_text = y['paragraph_text'].tolist()
            paragraph_spans = y['paragraph_tokens_spans'].tolist()

            if self.train:
                answer_start_span = torch.LongTensor(y['answer_start_span'].tolist())
                answer_end_span = torch.LongTensor(y['answer_end_span'].tolist())
                yield (
                    paragraph,
                    paragraph_word_features,
                    paragraph_features,
                    paragraph_mask,
                    question,
                    question_word_features,
                    question_mask,
                    answer_start_span,
                    answer_end_span,
                    paragraph_text,
                    paragraph_spans
                )
            else:
                yield (
                    paragraph,
                    paragraph_word_features,
                    paragraph_features,
                    paragraph_mask,
                    question,
                    question_word_features,
                    question_mask,
                    paragraph_text,
                    paragraph_spans
                )
