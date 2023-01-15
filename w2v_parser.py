import json
from time import time

import numpy
from gensim.models import KeyedVectors

from spacy.lang.en import English
import string


class Parser:
    def __init__(self, stop_words, nlp):
        self._tokenizer = nlp.tokenizer
        self._stopwords = stop_words

    def parse(self, query: str):
        query = query.translate(str.maketrans('', '', string.punctuation))
        doc_tokens = self._tokenizer(query)
        return list(filter(lambda tok: False if tok in self._stopwords else True, map(lambda tok: tok.lower_, doc_tokens)))


class QueryExpand:
    def __init__(self, stop_words, path_to_w2v_model):
        self.nlp = English()
        self._parser = Parser(stop_words, self.nlp)
        self.model = KeyedVectors.load(path_to_w2v_model)
        # self.model = KeyedVectors()

    def __call__(self, query, limiter):
        lst_of_tokens = self._parser.parse(query)
        expantion = []
        for token in lst_of_tokens:
            try:
                expantion.extend(self.model.most_similar(token, topn=5))
            except:
                print("Word not in w2v Dictionary")
        expantion.sort(key=lambda x: x[1], reverse=True)

        if len(expantion) >= limiter:
            return expantion[:limiter]

        return expantion


if __name__ == '__main__':
    with open('new_train_q.json', 'rb') as f:
        qu = json.load(f)
    ex = QueryExpand(('a'), "./w2v-models/word2vec.wordvectors")
    for q in qu:
        print(q)
        t = time()
        print(ex(q))
        print(time() - t)
