import json
from time import time

import numpy
from gensim.models import KeyedVectors

from spacy.lang.en import English
import string

"""
The Parser class is used to parse and process a given query string.
"""


class Parser:

    def __init__(self, stop_words, nlp):
        self._tokenizer = nlp.tokenizer
        self._stopwords = stop_words
    """
    Returns
    A list of tokens (words) that represent the query after being processed, lowercased and stop words removed.
    """
    def parse(self, query: str):
        query = query.translate(str.maketrans('', '', string.punctuation))
        doc_tokens = self._tokenizer(query)
        return list(
            filter(lambda tok: False if tok in self._stopwords else True, map(lambda tok: tok.lower_, doc_tokens)))


"""
The QueryExpand class is used to expand a given query by adding semantically similar terms to the query.
"""
class QueryExpand:
    def __init__(self, stop_words, path_to_w2v_model):
        self.nlp = English()
        self._parser = Parser(stop_words, self.nlp)
        self.model = KeyedVectors.load(path_to_w2v_model)

    """
    Returns
    A list of tuples where each tuple contains a word and its similarity score, sorted by similarity score in descending order.
    """
    def __call__(self, query, limiter):
        lst_of_tokens = self._parser.parse(query)
        expantion = []
        # for each token in the token list try to expand it
        for token in lst_of_tokens:
            try:
                expantion.extend(self.model.most_similar(token, topn=5))
            except:
                print("Word not in w2v Dictionary")
        expantion.sort(key=lambda x: x[1], reverse=True)

        if len(expantion) >= limiter:
            return expantion[:limiter]

        return expantion

