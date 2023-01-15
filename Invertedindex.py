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

# Let's start with a small block size of 30 bytes just to test things out.
# with open("Config.json", "r") as f:
#     config_dict = json.load(f)
#     index_config_dict = config_dict['index_configuration']

BLOCK_SIZE = 1999998
TUPLE_SIZE = 6
TF_MASK = 65535


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
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """

        with closing(MultiFileReader(self._base_dir)) as reader:
            for token in tokens:
                if token in self.posting_locs:
                    locs = self.posting_locs[token]
                    b = reader.read(locs, self.df[token] * TUPLE_SIZE)
                    # todo maybe implement using numpy array
                    posting_list = [None, ] * self.df[token]
                    for i in range(self.df[token]):
                        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                        tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                        posting_list[i] = (doc_id, tf)
                    yield token, posting_list



    def posting_lists_iter_filter(self,tokens:list, num:int):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader(self._base_dir)) as reader:
            for token in tokens:
                if token in  self.posting_locs:
                    locs = self.posting_locs[token]
                    b = reader.read(locs, self.df[token] * TUPLE_SIZE)
                    #todo maybe implement using numpy array

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
        # convert df to idf dictionary

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
