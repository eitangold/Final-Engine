from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Let's start with a small block size of 30 bytes just to test things out.
with open("Config.json","r") as f:
    config_dict = json.load(f)
    index_config_dict = config_dict['index_configuration']

BLOCK_SIZE = int(index_config_dict['BLOCK_SIZE'])
TUPLE_SIZE = int(index_config_dict['TUPLE_SIZE'])    
TF_MASK = int(index_config_dict['TF_MASK'])

class InvertedIndex:  
    def __init__(self):
        logger.critical("this is creating the inverted index")
        self.df = Counter()
        self.term_total = Counter()
        self.global_doc_size = dict()
        #maybe use numpy array 
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)
           

    def _write_globals(self, base_dir, name,bucket_name,path):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(f"postings_gcp/{path}{name}.pkl")
        blob_posting_locs.upload_from_filename(f"{base_dir}/{name}.pkl")


    def write_index(self, base_dir, name,bucket_name,path):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name,bucket_name,path)

    
    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def calculate_term_total():
         with closing(MultiFileReader()) as reader:    
            for token in self.posting_locs:
                locs = self.posting_locs[token]
                b = reader.read(locs, self.df[token] * TUPLE_SIZE)
                convert_to_int = lambda i,b: int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                self.term_total[token] = sum(conver_to_int(i,b) for i in range(self.df[token]))
              

    def posting_lists_iter(self,tokens:list):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader(self._base_dir)) as reader:
            for token in tokens:
                if token in  self.posting_locs:
                    locs = self.posting_locs[token]
                    b = reader.read(locs, self.df[token] * TUPLE_SIZE)
                    #todo maybe implement using numpy array
                    posting_list = [None,]*self.df[token]
                    for i in range(self.df[token]):
                        doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                        tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                        posting_list[i] = (doc_id, tf)
                    yield token, posting_list

    def search(q):
        pass
    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    # todo this function is called for each bucket that we have in the RDD -> more buckets -> more writing
    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name,stem,index_type):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        path = f"{stem}-index/{index_type}-index/"
        with closing(MultiFileWriter(".", bucket_id, bucket_name,stem,index_type,path)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big') for doc_id, tf in pl])
                locs = writer.write(b)
                posting_locs[w].extend(locs)
            writer.upload_to_gcp(path)
            InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name,stem,index_type)
        return bucket_id

    
    @staticmethod
    def _upload_posting_locs(bucket_id, posting_locs, bucket_name,stem,index_type):
        try:
            with open(f"{stem}_{index_type}_{bucket_id}_posting_locs.pickle", "wb") as f:
                pickle.dump(posting_locs, f)
        except:
              logger.critical("_upload_posting_locs thorw errorrrrrrr!!!!!")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        # todo blob_posting_locs = bucket.blob(f"postings_gcp/buckets/{bucket_id}_posting_locs.pickle")
        # todo latest ======= here path added 
        path = f"{stem}-index/{index_type}-index/"
        blob_posting_locs = bucket.blob(f"postings_gcp/{path}{stem}_{index_type}_{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{stem}_{index_type}_{bucket_id}_posting_locs.pickle")
    



class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name, bucket_name,stem,index_type,path):
        self._base_dir = Path(base_dir)
        self._path = path
        self._name = f"{stem}_{index_type}_{str(name)}"
        self._file_gen = (open(self._base_dir / f'{self._name}_{i:03}.bin', 'wb') 
                          for i in itertools.count())
        self._f = next(self._file_gen)
        # Connecting to google storage bucket. 
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
        # if the current file is full, close and open a new one.
            if remaining == 0:  
                self._f.close()
                self.upload_to_gcp(self._path)
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()
    
    def upload_to_gcp(self,path):
        '''
            The function saves the posting files into the right bucket in google storage.
        '''
        file_name = self._f.name
        blob = self.bucket.blob(f"postings_gcp/{path}{file_name}")
        blob.upload_from_filename(file_name)

        

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


