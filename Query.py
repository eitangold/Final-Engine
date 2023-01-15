import numpy.linalg

from factory import *
from collections import Counter


class Query:
    def __init__(self, tokenizer_name, stemmer_name):
        self.tf = Counter()
        self.query = None
        self.tokenizer = FactoryIndex.get_tokenizer(tokenizer_name)
        self._stemmer = FactoryIndex.get_stemmer(stemmer_name)
        self.norm = 0

    def __call__(self, query, stopwords_frozen, ranking_type=None, normalize_tf=None, idf_dict=None):
        if isinstance(query, str):
            RE_WORD = self.tokenizer
            list_of_tokens_q = []
            # do stemmer and calculate _tf
            if self._stemmer is not None:
                list_of_tokens_q = [self._stemmer.stem(token.group()) for token in RE_WORD.tokenize(query.lower()) if
                                    token.group() not in stopwords_frozen]
            else:
                list_of_tokens_q = [token.group() for token in RE_WORD.tokenize(query.lower()) if
                                    token.group() not in stopwords_frozen]
        elif isinstance(query, list):
            list_of_tokens_q = [self._stemmer.stem(token) for token in query]

        self.tf = Counter(list_of_tokens_q)
        if ranking_type == 'binary_ranking':
            pass
        else:
            try:
                normalizer = 1
                if normalize_tf == 'norm_len':
                    normalizer = len(list_of_tokens_q)

                set_of_tokens = set(list_of_tokens_q)
                for term in set_of_tokens:
                    try: 
                        self.tf[term] /= normalizer
                        self.tf[term] *= idf_dict[term]
                    except:
                        print(self.tf.keys())
                        print("word not in dictionary")
                        self.tf.pop(term)
                self.norm = numpy.linalg.norm(numpy.array(list(self.tf.values())))
            except:
                pass


