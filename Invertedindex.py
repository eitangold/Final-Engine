import json
import logging
import math
import os.path
import pickle
from collections import Counter
# from google.cloud import storage
from collections import defaultdict
from contextlib import closing
# import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
Notes this is where all the global variabels lay
"""
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6
TF_MASK = 65535
"""
idf: a dictionary to store the Inverse Document Frequency (IDF) values for each term.
_base_dir: a variable to store the base directory path for the documents.
df: a counter to keep track of the Document Frequency (DF) of each term.
term_total: a counter to keep track of the total number of occurrences of each term in the collection.
_posting_list: a defaultdict to store the posting list for each term, where the key is the term and the value is a list of documents that contain the term.
posting_locs: a defaultdict to store the location of each term in the documents, where the key is the term and the value is a list of tuples, each representing the document ID and the location of the term in the document.
_doc_to_len: a dictionary to store the length of each document.
_doc2stat: a dictionary to store the statistics of each document.
_doc2norm: a dictionary to store the normalization values of each document.
_N: a variable to store the total number of documents in the collection.
_avg: a variable to store the average length of the documents in the collection.
"""

class InvertedIndex:
    def __init__(self):
        self.idf = {}
        self._base_dir = None
        self.df = Counter()
        self.term_total = Counter()
        # self.global_doc_size = dict()
        # maybe use numpy array
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)
        self._doc_to_len = {}
        self._doc2stat = {}
        self._doc2norm = {}
        self._N = 0
        self._avg = 0


    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def calculate_term_total(self):
        with closing(MultiFileReader(self._base_dir)) as reader:
            convert_to_int = lambda i, b: int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            for token in self.posting_locs:
                locs = self.posting_locs[token]
                b = reader.read(locs, self.df[token] * TUPLE_SIZE)
                self.term_total[token] = sum(convert_to_int(i, b) for i in range(self.df[token]))

    def posting_lists_iter(self, tokens: list):
        """
        A generator that reads one posting list from disk and yields  a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """

        with closing(MultiFileReader(self._base_dir)) as reader:
            for token in tokens:
                if token in self.posting_locs:
                    locs = self.posting_locs[token]
                    b = reader.read(locs, self.df[token] * TUPLE_SIZE)
                    posting_list = [None, ] * self.df[token]
                    for i in range(self.df[token]):
                        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                        tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                        posting_list[i] = (doc_id, tf)
                    yield token, posting_list



    def posting_lists_iter_filter(self,tokens:list, num:int):
        """
        The posting_lists_iter_filter method is a generator function that allows for iterating over the posting lists
        of a given set of terms, while also filtering the number of documents to be returned for each term.
        """
        with closing(MultiFileReader(self._base_dir)) as reader:
            for token in tokens:
                if token in  self.posting_locs:
                    locs = self.posting_locs[token]
                    b = reader.read(locs, self.df[token] * TUPLE_SIZE)
                    if num != 0:
                        stop = num if num <= self.df[token] else self.df[token]
                    else:
                        stop = self.df[token]
                    posting_list = [None,]*stop
                    for i in range(stop):
                        doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                        tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                        posting_list[i] = (doc_id, tf)
                    yield token, posting_list

    def warm_up(self):
        """
            The warm_up method is used to pre-compute some values that will be used later in the retrieval process.
            These pre-computed values will be used later in the retrieval process to improve the efficiency of the retrieval process.
        """
        self._N = len(self._doc_to_len)
        for word, df in self.df.items():
            self.idf[word] = math.log2(self._N / df)

        count_lens = sum(self._doc_to_len.values())
        self._avg = count_lens / self._N

    @staticmethod
    def read_index(base_dir, name):
        path = Path(base_dir) / f'{name}.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                idx = pickle.load(f)
                idx._base_dir = base_dir
                idx.idf = {}
                return idx
        else:
            raise FileExistsError


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self, basedir):
        self._open_files = {}
        self._basedir = basedir

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(Path(self._basedir) / f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
