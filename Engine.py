import json
import os.path
import pickle
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

from Invertedindex import *
from Query import *
from SimilariyFunctions import cosine_similarity, binary_rank, BM25_score

FULL_POWER = 'FALSE'


class Engine:
    def __init__(self, engine_config_file) -> None:

        def init_index(index: InvertedIndex, files_to_read):
            if 'word2df' in files_to_read:
                path = files_to_read['word2df']
                print(f"trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as w2df:
                        index.df = pickle.load(w2df)
            if 'doc2stat' in files_to_read:
                path = files_to_read['doc2stat']
                print(f"trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2st:
                        index._doc2stat = pickle.load(d2st)
            if 'doc2len' in files_to_read:
                path = files_to_read['doc2len']
                print(f"trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2size:
                        index._doc_to_len = pickle.load(d2size)
            if 'doc2norm' in files_to_read:
                path = files_to_read['doc2norm']
                print(f"trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2norm:
                        index._doc2norm = pickle.load(d2norm)

        # todo handel page rank
        # TODO wrap all the loading in try except

        # TODO change the index to the one we choose
        self._doc2title = {}
        self._page_rank = {}
        self._doc2page_view = {}
        self._cache = Counter()

        with open(engine_config_file, "r") as f:
            engine_indices_config = json.load(f)
            regular_index_config = engine_indices_config['regular']

            if 'snowball' in engine_indices_config:
                print(f"trying to create snowball index")
                our_index_config = engine_indices_config['snowball']
            elif 'porter' in engine_indices_config:
                print(f"trying to create porter index")
                our_index_config = engine_indices_config['porter']
        """
         this part sets all the dictionary's that relative to the Engine obj
         each dictionary that is common to all the indices is in the same folder as the engine
        """
        if FULL_POWER == 'TRUE':
            # with open(engine_config_file['path_to_page_view'], 'rb') as f:
            #     print(f"trying to create {engine_config_file['path_to_page_view']}")
            #     self._doc2page_view = pickle.load(f)
            # with open(engine_config_file['path_to_page_rank'], 'rb') as f:
            #     print(f"trying to create {engine_config_file['path_to_page_rank']}")
            #     self._page_rank = pickle.load(f)
            with open(engine_indices_config['path_to_doc2title'], 'rb') as f:
                print(f"trying to create {engine_indices_config['path_to_doc2title']}")
                self._doc2title = pickle.load(f)

        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self._stopwords = english_stopwords.union(corpus_stopwords)
        ### full power represent to unleash the full power and load it to memory :) ###
        if FULL_POWER == 'FALSE':
            ########## initialize only low power amount of index 1-2 maybe  (test = must have)   (best = Ours) ##########
            try:
                reg_idx_txt_cnfg = regular_index_config['text']
                print(f"#### trying to create text index ######")
                self._index_text = InvertedIndex.read_index(reg_idx_txt_cnfg['path_to_index_base_dir'],
                                                            reg_idx_txt_cnfg['index_name'])
                init_index(self._index_text, reg_idx_txt_cnfg)
                self._index_text.warm_up()
            except Exception:
                print("######Erorr while loading the small power index :((((((########")

        else:
            ########## initialize all the indices (test = must have)   (best = Ours) ##########
            reg_idx_txt_cnfg = regular_index_config['text']
            reg_idx_title_cnfg = regular_index_config['title']
            reg_idx_anchor_cnfg = regular_index_config['anchor']
            try:
                self._index_text = InvertedIndex.read_index(reg_idx_txt_cnfg['path_to_index_base_dir'],
                                                            reg_idx_txt_cnfg['index_name'])
                init_index(self._index_text, reg_idx_txt_cnfg)
            except Exception:
                print("Error while loading the regular text index")

            try:
                self._index_title = InvertedIndex.read_index(reg_idx_title_cnfg['path_to_index_base_dir'],
                                                             reg_idx_title_cnfg['index_name'])
                init_index(self._index_title, reg_idx_title_cnfg)
            except Exception:
                print("Error while loading the regular title index")
            try:
                self._index_anchor = InvertedIndex.read_index(reg_idx_anchor_cnfg['path_to_index_base_dir'],
                                                              reg_idx_anchor_cnfg['index_name'])
                init_index(self._index_anchor, reg_idx_anchor_cnfg)
            except Exception:
                print("Error while loading the regular anchor index")
                ###### warming up the engine in the cold cold winter #####
            try:
                self._index_text.warm_up()
            except Exception:
                print("Error while warming up the index")

            ########## this is the part of ours index ##########
            try:
                our_idx_text_cnfg = our_index_config['text']
                self._best_text_index = InvertedIndex.read_index(our_idx_text_cnfg['path_to_index_base_dir'],
                                                                 our_idx_text_cnfg['index_name'])
                init_index(self._best_text_index, our_idx_text_cnfg)
            except Exception:
                print("Error while loading the our text index")
            try:
                our_idx_title_cnfg = our_index_config['title']
                self._best_title_index = InvertedIndex.read_index(our_idx_title_cnfg['path_to_index_base_dir'],
                                                                  our_idx_title_cnfg['index_name'])
                init_index(self._best_title_index, our_idx_title_cnfg)
            except Exception:
                print("Error while loading the out title index")

    def search(self, query, type_of_stemmer):
        # q = Query('regex', 'regular')
        # q(query, self._stopwords)
        # # if self.filter is None:
        # posting_list = self._index_text.posting_lists_iter(list(q.tf.keys()))
        # # else:
        # #   posting_list = self._index_text.posting_lists_iter(list(q.tf.keys()), self.filter)
        # result = BM25_score(q, posting_list, 0.75, 1.2, 10, self._index_text._doc_to_len, self._index_text.idf,
        #                     self._index_text._avg)
        # a = self.get_top_n(result, 100)
        # return a
        pass

    def search_body(self, query):
        q = Query('regex', 'regular')
        q(query=query, stopwords_frozen=self._stopwords, normalize_tf="norm_len", idf_dict=self._index_text.idf)
        posting_list = self._index_text.posting_lists_iter(list(q.tf.keys()))

        result = cosine_similarity(q, posting_list, self._index_text.idf, self._index_text._doc_to_len,
                                   self._index_text._doc2norm)
        return self.get_top_n_with_title(result, 100)

    def search_title(self, query):
        q = Query('regex', 'regular')
        q(query, self._stopwords, ranking_type="binary_ranking")

        posting_list = self._index_title.posting_lists_iter(list(q.tf.keys()))

        result = binary_rank(posting_list)
        return self.get_top_n_with_title(result, 100)

    def search_anchor(self, query):
        q = Query('regex', 'regular')
        q(query, self._stopwords, ranking_type="binary_ranking")

        posting_list = self._index_anchor.posting_lists_iter(list(q.tf.keys()))

        result = binary_rank(posting_list)
        return self.get_top_n_with_title(result, 100)

    def get_pagerank(self, lst_of_pages):
        return [self._page_rank[doc_id] for doc_id in lst_of_pages]

    def get_pageview(self, lst_of_pages):
        return [self._doc2page_view[doc_id] for doc_id in lst_of_pages]

    def get_top_n(self, sim_dict, N=3):
        # return sorted([(doc_id,round(score,5),self._doc2title[doc_id]) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]
        return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]

    def get_top_n_with_title(self, sim_dict, N=3):
        # return sorted([(doc_id,round(score,5),self._doc2title[doc_id]) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]
        top_100 = sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]
        return list(map(lambda x: (x[0], self._doc2title[x[0]]), top_100))

    def test_engine(self, true_list, predicted_list):
        def average_precision(true_list, predicted_list, k=40):
            true_set = frozenset(true_list)
            predicted_list = predicted_list[:k]
            precisions = []
            for i, doc_id in enumerate(predicted_list):
                if doc_id in true_set:
                    prec = (len(precisions) + 1) / (i + 1)
                    precisions.append(prec)
            if len(precisions) == 0:
                return 0.0
            return round(sum(precisions) / len(precisions), 3)

        return average_precision(true_list, predicted_list)

    def get_words(self):
        return self._index_text._posting_list.keys()

    def check_if_path_exists(self, path):
        return os.path.exists(path)
