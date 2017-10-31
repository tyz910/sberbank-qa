import os
import re
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.util import align_tokens
import pymorphy2
import pandas as pd
import collections
from tqdm import *


class Lemma(object):
    def __init__(self, lemma, pos, embedding_idx):
        self.lemma = lemma
        self.pos = pos
        self.embedding_idx = embedding_idx

    def __repr__(self):
        return self.lemma.__repr__()


class Vocab(object):
    GRAMMAR_MAP = {
        'NOUN': ['_NOUN', '_ADJ', '_VERB', '_ADV'],
        'VERB': ['_VERB', '_NOUN', '_ADJ', '_ADV'],
        'INFN': ['_VERB', '_NOUN', '_ADJ', '_ADV'],
        'GRND': ['_VERB', '_NOUN', '_ADJ', '_ADV'],
        'PRTF': ['_VERB', '_NOUN', '_ADJ', '_ADV'],
        'PRTS': ['_VERB', '_NOUN', '_ADJ', '_ADV'],
        'ADJF': ['_ADJ', '_NOUN', '_ADV'],
        'ADJS': ['_ADJ', '_NOUN', '_ADV'],
        'ADVB': ['_ADV', '_NOUN'],
        'CONJ': ['_CCONJ', '_ADV'],
        'COMP': ['_ADV', '_ADP'],
        'PREP': ['_ADP', '_ADV'],
        'PRCL': ['_PART', '_ADV'],
        'INTJ': ['_INTJ', '_NOUN'],
        'PRED': ['_ADV'],
        'NPRO': ['_PRON'],
        'NUMR': ['_NUM'],
        'None': ['_NOUN']
    }

    def __init__(self, embedding_size=300):
        self.wv = None
        self.lemmas = {}
        self.tune_num = 0
        self.embedding_size = embedding_size
        self.embeddings = [
            [0] * self.embedding_size,  # PAD
            [0] * self.embedding_size   # UNK
        ]

    def rebuild(self, tune_lemmas):
        max_idx = len(self.embeddings)
        self.tune_num = len(tune_lemmas)

        old_embeddings = self.embeddings
        self.embeddings = self.embeddings[:2]
        for lemma in self.lemmas.values():
            if 1 < lemma.embedding_idx < max_idx:
                lemma.embedding_idx += max_idx

        for tune_lemma in tune_lemmas:
            if tune_lemma.embedding_idx > max_idx:
                embedding = old_embeddings[tune_lemma.embedding_idx - max_idx]
            else:
                embedding = old_embeddings[tune_lemma.embedding_idx]

            tune_lemma.embedding_idx = len(self.embeddings)
            self.embeddings.append(embedding)

        for lemma in self.lemmas.values():
            if lemma.embedding_idx > max_idx:
                embedding = old_embeddings[lemma.embedding_idx - max_idx]
                lemma.embedding_idx = len(self.embeddings)
                self.embeddings.append(embedding)

        del old_embeddings

    def shrink(self, embeddings):
        self.tune_num = 0
        self.embeddings = embeddings
        self.lemmas = {k: v for k, v in self.lemmas.items() if (v.embedding_idx < len(embeddings)) and (v.embedding_idx != 1)}

    def get_lemma(self, word_form):
        lemma = word_form.normal_form
        pos = str(word_form.tag.POS)

        if (lemma, pos) not in self.lemmas:
            embedding = self.__find_embedding(lemma, pos)
            if embedding:
                embedding_idx = len(self.embeddings)
                self.embeddings.append(embedding)
            else:
                embedding_idx = 1

            l = Lemma(lemma, pos, embedding_idx)
            self.lemmas[(lemma, pos)] = l

            return l
        else:
            return self.lemmas[(lemma, pos)]

    def __find_embedding(self, lemma, pos):
        if self.wv is None:
            wv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'wv_' + str(self.embedding_size) + '.pickle'))
            self.wv = pd.read_pickle(wv_path)

        if pos in Vocab.GRAMMAR_MAP:
            lemma = lemma.replace('ё', 'е')
            for wv_pos in Vocab.GRAMMAR_MAP[pos]:
                wv_lemma = lemma + wv_pos
                if wv_lemma in self.wv.index:
                    return self.wv.loc[wv_lemma].tolist()

        return None


class Token(object):
    GRAMMEMES = [
        '1per', '2per', '3per', 'ADJF', 'ADJS', 'ADVB', 'ANim', 'Abbr', 'Adjx', 'Af-p', 'Anph', 'Anum', 'Apro',
        'COMP', 'CONJ', 'Dmns', 'Fixd', 'GNdr', 'GRND', 'Geox', 'INFN', 'INTJ', 'Impe', 'Impx', 'Infr', 'Inmx',
        'LATN', 'Ms-f', 'NOUN', 'NPRO', 'NUMB', 'NUMR', 'Name', 'Orgn', 'PNCT', 'PRCL', 'PRED', 'PREP', 'PRTF',
        'PRTS', 'Patr', 'Pltm', 'Poss', 'Prdx', 'Prnt', 'Qual', 'Ques', 'ROMN', 'Sgtm', 'Subx', 'Supr', 'Surn',
        'UNKN', 'V-be', 'V-ej', 'V-sh', 'VERB', 'Vpre', 'ablt', 'accs', 'actv', 'anim', 'datv', 'excl', 'femn',
        'futr', 'gen2', 'gent', 'impf', 'impr', 'inan', 'indc', 'intg', 'intr', 'loc2', 'loct', 'masc', 'neut',
        'nomn', 'past', 'perf', 'plur', 'pres', 'pssv', 'real', 'sing', 'tran', 'voct'
    ]
    GRAMMEMES_MAP = {g: i for i, g in enumerate(GRAMMEMES)}

    def __init__(self, word, word_lower, stem, lemma, word_form):
        self.word = word
        self.word_lower = word_lower
        self.stem = stem
        self.lemma = lemma
        self._word_tag = word_form.tag
        self._tags = None

    def get_tags(self):
        if self._tags is None:
            self._tags = [g in self._word_tag for g in Token.GRAMMEMES]
            self._word_tag = None

        return self._tags

    def get_number(self):
        if self.is_grammeme(['NUMB']):
            try:
                numb = float(self.lemma.lemma.replace(',', '.'))
                return numb
            except ValueError:
                return 0
        else:
            return 0

    def is_grammeme(self, grammemes):
        tags = self.get_tags()
        for g in grammemes:
            if tags[Token.GRAMMEMES_MAP[g]]:
                return True
        return False

    def __repr__(self):
        return self.word.__repr__()


class Tokenizer(object):
    def __init__(self, vocab=None, morph=None, stemmer=None):
        self.vocab = vocab if vocab is not None else Vocab()
        self.morph = morph if morph is not None else pymorphy2.MorphAnalyzer()
        self.stemmer = stemmer if stemmer is not None else SnowballStemmer('russian')
        self.tokens = {}

    def tokenize(self, text):
        return [self.__get_token(word) for word in word_tokenize(text)]

    def reinit(self, morph=None):
        self.morph = morph if morph is not None else pymorphy2.MorphAnalyzer()
        self.vocab.wv = None

    def __get_token(self, word):
        if word not in self.tokens:
            parse_word = re.sub(r'([0-9]+)-\w', r'\g<1>', word)
            parse_word = parse_word.replace('\u0301', '')
            parse_word_parts = parse_word.split('-')
            parse_word_parts.reverse()
            if len(parse_word_parts) > 1:
                parse_word_parts = [parse_word] + parse_word_parts

            word_form = None
            lemma = None

            for part in parse_word_parts:
                if lemma is None:
                    word_form = self.morph.parse(part)[0]
                    lemma = self.vocab.get_lemma(word_form)
                elif lemma.embedding_idx == 1:
                    part_word_form = self.morph.parse(part)[0]
                    part_lemma = self.vocab.get_lemma(part_word_form)
                    if part_lemma.embedding_idx > 1:
                        word_form = part_word_form
                        lemma = part_lemma

            word_lower = word.lower()
            stem = self.stemmer.stem(word_lower)
            self.tokens[word] = Token(word, word_lower, stem, lemma, word_form)

        return self.tokens[word]


class Doc(object):
    def __init__(self, text, tokenizer=None):
        if tokenizer is None:
            tokenizer = Tokenizer()

        self.text = Doc.clean_text(text)
        self.tokens = tokenizer.tokenize(self.text)
        self.tokens_spans = None
        self.word_features = None

    def align_tokens(self):
        if self.tokens_spans is None:
            self.tokens_spans = align_tokens([t.word for t in self.tokens], self.text)

        return self.tokens_spans

    def get_embeddings(self):
        return [t.lemma.embedding_idx for t in self.tokens]

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\[([0-9]+|уточнить)\]|…|„|“', '', text)
        text = re.sub(r' (\.|,|\)|:)', r'\g<1>', text)
        text = re.sub(r'(\() ', r'\g<1>', text)
        text = re.sub(r'([0-9])—([0-9])', r'\g<1> — \g<2>', text)
        return text

    def get_word_features(self):
        if self.word_features is None:
            self.word_features = [
                [
                    t.get_number()
                ] + t.get_tags() for t in self.tokens
            ]

        return self.word_features

    def __getitem__(self, key):
        return self.tokens[key]

    def __repr__(self):
        return self.text.__repr__()


class Paragraph(Doc):
    def __init__(self, text, tokenizer):
        super().__init__(text, tokenizer)
        self.align_tokens()

    def find_answer_span(self, answer_start, answer_end):
        answer_start_span = None
        answer_end_span = None

        for i, span in enumerate(self.tokens_spans):
            if answer_start_span is None:
                if (span[0] <= answer_start) and (span[1] > answer_start):
                    answer_start_span = i
                    answer_end_span = i
            elif span[0] < answer_end:
                answer_end_span = i
            else:
                break

        return answer_start_span, answer_end_span


class Question(Doc):
    pass


class ParagraphQuestion(object):
    def __init__(self, paragraph, question, answer=None, answer_pos=0):
        self.paragraph = paragraph
        self.question = question
        self.answer = None
        self.answer_start = None
        self.answer_end = None
        self.answer_start_span = None
        self.answer_end_span = None

        if answer is not None:
            self.answer = Doc.clean_text(answer)
            self.answer = re.sub(r'\[[0-9]*$', '', self.answer)
            self.answer = self.answer.strip(' .?')

            matches = [m.start() for m in re.finditer(re.escape(self.answer.lower()), self.paragraph.text.lower())]
            self.answer_start = matches[answer_pos]
            self.answer_end = self.answer_start + len(self.answer)
            self.answer_start_span, self.answer_end_span = self.paragraph.find_answer_span(self.answer_start, self.answer_end)

    def get_paragraph_features(self):
        question_words = {t.word_lower for t in self.question.tokens if not t.is_grammeme(['PNCT'])}
        question_stems = {t.stem for t in self.question.tokens if not t.is_grammeme(['PNCT'])}
        question_lemmas = {t.lemma.lemma for t in self.question.tokens if not t.is_grammeme(['PNCT'])}

        counter = collections.Counter(t.lemma.lemma for t in self.paragraph.tokens if not t.is_grammeme(['PNCT', 'PREP', 'CONJ']))
        total = sum(counter.values())

        return [
            [
                counter[t.lemma.lemma] / total,
                t.word_lower in question_words,
                t.stem in question_stems,
                t.lemma.lemma in question_lemmas
            ] for t in self.paragraph.tokens
        ]

    def __repr__(self):
        return [self.paragraph, self.question, self.answer].__repr__()


class QA(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.paragraphs = {}
        self.questions = {}
        self.paragraph_questions = []

    def add_paragraph(self, text, id=None):
        if id is None:
            id = text

        paragraph = Paragraph(text, self.tokenizer)
        self.paragraphs[id] = paragraph
        return paragraph

    def get_paragraph(self, id):
        if id in self.paragraphs:
            return self.paragraphs[id]
        else:
            return None

    def add_question(self, text, id=None):
        if id is None:
            id = text

        question = Question(text, self.tokenizer)
        self.questions[id] = question
        return question

    def get_question(self, id):
        if id in self.questions:
            return self.questions[id]
        else:
            return None

    def add(self, paragraph, question, answer=None, answer_pos=0):
        if not isinstance(paragraph, Paragraph):
            id = paragraph
            paragraph = self.get_paragraph(id)
            if paragraph is None:
                paragraph = self.add_paragraph(id)

        if not isinstance(question, Question):
            id = question
            question = self.get_question(id)
            if question is None:
                question = self.add_question(id)

        qa = ParagraphQuestion(paragraph, question, answer, answer_pos)
        self.paragraph_questions.append(qa)

        return qa

    def load_df(self, df, answers=True, answer_pos_df=None):
        for i, row in tqdm(df.iterrows(), total=len(df), unit='doc'):
            paragraph_id = row['paragraph_id']
            paragraph = self.get_paragraph(paragraph_id)
            if paragraph is None:
                paragraph = self.add_paragraph(row['paragraph'], paragraph_id)

            question_id = row['question_id']
            question = self.get_question(question_id)
            if question is None:
                question = self.add_question(row['question'], question_id)

            if answers:
                answer = row['answer']
            else:
                answer = None

            answer_pos = 0
            if answer_pos_df is not None:
                if i in answer_pos_df.index:
                    answer_pos = int(answer_pos_df.loc[i]['answer_pos'])

            self.add(paragraph, question, answer, answer_pos)

    def load_train(self, head=None, max_answers=None):
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv'))
        df = pd.read_csv(csv_path)
        df.drop([30432, 34157], inplace=True)

        answer_pos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'answer_pos.csv'))
        answer_pos_df = pd.read_csv(answer_pos_path, index_col=0)

        if max_answers is not None:
            count_answers = df.apply(
                lambda r: len([m.start() for m in re.finditer(re.escape(r['answer'].lower()), r['paragraph'].lower())]),
                axis=1
            )
            df = df[count_answers <= max_answers]

        if head is not None:
            df = df.head(head)

        self.load_df(df, answer_pos_df=answer_pos_df)
        return df

    def docs(self):
        for doc in self.questions.values():
            yield doc

        for doc in self.paragraphs.values():
            yield doc

    def count_lemmas(self, docs=None):
        if docs is None:
            docs = self.docs()

        counter = collections.Counter(t.lemma for doc in docs for t in doc.tokens)

        data = []
        for l, cnt in counter.items():
            data.append([l, cnt, l.embedding_idx > 1])

        df = pd.DataFrame(data, columns=['lemma', 'cnt', 'embedding'])
        df.set_index('lemma', inplace=True)

        return df.sort_values('cnt', ascending=False)

    def rebuild_vocab(self, tune_num=200):
        c = self.count_lemmas()

        def filter_lemmas(l):
            tag = self.tokenizer.morph.parse(l.lemma)[0].tag
            for g in ['ADJF', 'ADVB', 'CONJ', 'INFN', 'NPRO', 'PRED', 'PREP', 'PRCL', 'PNCT', 'ROMN', 'LATN', 'UNKN']:
                if g in tag:
                    return True

            return False

        c = c[c['embedding'] == False]
        c = c[list(c.index.map(filter_lemmas))]
        self.tokenizer.vocab.rebuild(c.head(tune_num).index.tolist())

    def get_Xy(self, tqdm_show=True):
        X = []
        y = []

        paragraph_questions = self.paragraph_questions
        if tqdm_show:
            paragraph_questions = tqdm(paragraph_questions, total=len(paragraph_questions), unit='doc')

        for qa in paragraph_questions:
            X.append([
                qa.paragraph.get_embeddings(),
                qa.paragraph.get_word_features(),
                qa.get_paragraph_features(),
                qa.question.get_embeddings(),
                qa.question.get_word_features()
            ])

            y.append([
                qa.answer,
                qa.answer_start_span,
                qa.answer_end_span,
                qa.paragraph.text,
                qa.paragraph.tokens_spans
            ])

        X_df = pd.DataFrame(X, columns=[
            'paragraph_embeddings',
            'paragraph_word_features',
            'paragraph_features',
            'question_embeddings',
            'question_word_features'
        ])

        y_df = pd.DataFrame(y, columns=[
            'answer',
            'answer_start_span',
            'answer_end_span',
            'paragraph_text',
            'paragraph_tokens_spans'
        ])

        return X_df, y_df

    def __getitem__(self, key):
        return self.paragraph_questions[key]

    def __repr__(self):
        return self.paragraph_questions.__repr__()
