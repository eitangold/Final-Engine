import json
import os.path
import pickle
from time import time

import nltk

import factory

nltk.download('stopwords')
from nltk.corpus import stopwords
from multiprocessing import Process, Queue
from Invertedindex import *
from Query import *
from w2v_parser import *
from SimilariyFunctions import cosine_similarity, binary_rank, BM25_score
import pandas as pd
FULL_POWER = 'TRUE'

"""
this class is the search engine that retrieve the query's
the search engine works as pub and sub 
"""

class Engine:

    def multi_process_index(self, inputQ: Queue,index_config:dict,indextype,outputQ,stop_words_set):
        """"
          :argument inputQ-Q that the main process thorw jobs at each index(query) 
          :argument index_config the configuraion for each index ( word2vec , stemmer , where the location all the files
          :argument outputQ after the process (index) has retrived the doc related to the query this is the main pipe that all the process communicate with 
          :argument stop_words_set - is the stopword set that each process will get
          :return this function does not return since its a process so the process puts the answer in the outputQ
        """""
        try:
            """
             this is the part where each index load all the data to himsel  
            """
            index = InvertedIndex.read_index(index_config['path_to_index_base_dir'],index_config['index_name'])
            if 'word2df' in index_config:
                path = index_config['word2df']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as w2df:
                        index.df = pickle.load(w2df)
                else:
                     print(f"######## {indextype} {path} didnt found #######")

            if 'doc2len' in index_config:
                path = index_config['doc2len']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2size:
                        index._doc_to_len = pickle.load(d2size)
                else:
                     print(f"######## {indextype} {path} didnt found #######")
            if 'doc2norm' in index_config:
                path = index_config['doc2norm']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2norm:
                        index._doc2norm = pickle.load(d2norm)
                else:
                     print(f"######## {indextype} {path} didnt found #######")
            if 'page-rank' in index_config:
                path = index_config['page-rank']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path,'rb') as f:
                        df_page_rank = pickle.load(f)
                else:
                     print(f"######## {indextype} {path} didnt found #######")
            if 'word2vec' in index_config:
                path = index_config['word2vec']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    # creating the word2vec model also the Query
                    expander = QueryExpand(stop_words_set,path)
                    q = Query('regex', 'snowball')

            if indextype == "regular-text-index" or indextype == "our-text-index":
                print(" ### trying to warm up text index ##### ")
                index.warm_up()
                print(" ### finished to warm up text index ##### ")
            if indextype in ["regular-text-index","regular-title-index","regular-anchor-index"]:
                q = Query('regex', 'regular')

        except Exception:
            print(f"Error while loading the {indextype} text index")
            exit(999)

        while True:
            """
             the engine communicate with the process through a Queue and each process is waiting for job listening to the pipe
             the output from the pipe is the input for the process and its the dada that to process needs
             :var message is the message with all the job details that the process need to start working
             :var is the variable that set the threshold for the word2vec model on how much words he will expend the query
             :function posting_list_iter_filter filter the amount of docs the index will retrieve
            """
            output = []
            message = inputQ.get()
            query = message[0]
            type_of_search = message[1]
            type_of_index = message[2]
            if type_of_index == "our-index":
                limiter = message[3]
                # this parameter determain if we use word2vec model or not
                w2v_on_of = message[4]
                # this is the amount of weight we put on the model
                weight = message[5]
                if w2v_on_of == "on":
                    expanded_query = list(map(lambda tup: tup[0],expander(query,limiter)))
                    q1_words = q(query=expanded_query, stopwords_frozen=stop_words_set, normalize_tf=None, idf_dict=index.idf)
                    posting_list = index.posting_lists_iter_filter(q1_words,200000)
                    ##  this is where we cosine/BM25
                    output = BM25_score(posting_list, index.idf, index._doc_to_len, index._avg)
                    if len(output.keys()) > 0:
                        normelize_factor = output.most_common(1)[0][1]
                        for key in output:
                            output[key] = output.get(key,0)*weight*(1/normelize_factor)

                    output = Counter(dict(output.most_common(3000)))

                else:
                    q = Query('regex', 'snowball')
                    q1_words = q(query=query, stopwords_frozen=stop_words_set, normalize_tf=None, idf_dict=index.idf)
                    posting_list = index.posting_lists_iter_filter(q1_words,200000)
                    ##  this is where we cosine/BM25
                    output = BM25_score(posting_list, index.idf, index._doc_to_len, index._avg)
                    if len(output.keys()) > 0:
                        normelize_factor = output.most_common(1)[0][1]
                        for key in output:
                            output[key] = output.get(key, 0) * weight * (1 / normelize_factor)
                    output = Counter(dict(output.most_common(3000)))


            # regular index is the index that is used  to answer querys that are not from out search function
            # our parameter represet an index of our
            elif type_of_index == "regular-index":
                if type_of_search == "cosine":
                    q1_words = q(query=query, stopwords_frozen=stop_words_set, normalize_tf="norm_len", idf_dict=index.idf)
                    posting_list = index.posting_lists_iter(q1_words)
                    ##  this is where we cosine/BM25
                    output = cosine_similarity(q, posting_list,index.idf, index._doc_to_len,index._doc2norm)
                elif type_of_search == 'binary':
                    if message[3] == "our":
                        weight = message[4]
                        q1_words = q(query=query, stopwords_frozen=stop_words_set, normalize_tf=None, idf_dict=index.idf,ranking_type="binary_ranking")
                        ##  this is where we binary
                        posting_list = index.posting_lists_iter(q1_words)
                        output = binary_rank(posting_list)
                        if len(output.keys()) > 0:
                            normelize_factor = len(q1_words)
                            for key in output:
                                output[key] = (output.get(key, 0) * weight * (1 / normelize_factor))*0.8 + (df_page_rank.get(key,0)/9914)*0.2


                        output = Counter(dict(output.most_common(3000)))
                        indextype = "special"

                    else:
                        q1_words = q(query=query, stopwords_frozen=stop_words_set, normalize_tf=None, idf_dict=index.idf,ranking_type="binary_ranking")
                        ##  this is where we binary
                        posting_list = index.posting_lists_iter(q1_words)
                        output = binary_rank(posting_list)

                elif type_of_search == "page-rank":
                    lst_of_pages = query
                    output = []
                    for i in lst_of_pages:
                        output.append(df_page_rank.get(i,0))

            outputQ.put((output, indextype))

    """"
    :argument inputQ-Q that the main process thorw jobs at each index(query) 
    :argument index_config the configuraion for each index ( word2vec , stemmer , where the location all the files
    :argument outputQ after the process (index) has retrived the doc related to the query this is the main pipe that all the process communicate with 
    :argument stop_words_set - is the stopword set that each process will get
    :return this function does not return since its a process so the process puts the answer in the outputQ
    """""
    def one_core_regular_anchor(self, inputQ: Queue,index_config:dict,indextype,outputQ,stop_words_set):
        try:
            index = InvertedIndex.read_index(index_config['path_to_index_base_dir'], index_config['index_name'])
            if 'word2df' in index_config:
                path = index_config['word2df']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as w2df:
                        index.df = pickle.load(w2df)
                else:
                    print(f"######## {indextype} {path} didnt found #######")

            if 'doc2len' in index_config:
                path = index_config['doc2len']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2size:
                        index._doc_to_len = pickle.load(d2size)
                else:
                    print(f"######## {indextype} {path} didnt found #######")
            if 'doc2norm' in index_config:
                path = index_config['doc2norm']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2norm:
                        index._doc2norm = pickle.load(d2norm)
                else:
                    print(f"######## {indextype} {path} didnt found #######")
            if 'page-rank' in index_config:
                path = index_config['page-rank']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path,'rb') as f:
                        df_page_rank = pickle.load(f)
                    print("##finished reading page rank anchor")
                else:
                    print(f"######## {indextype} {path} didnt found #######")
            q = Query('regex', 'regular')

        except Exception:
            print(f"Error while loading the {indextype} text index")
            exit(999)

        while True:
            # waiting to receive message(query)
            message = inputQ.get()
            query = message[0]
            type_of_index = message[1]
            #parsing the query
            q1_words = q(query=query, stopwords_frozen=stop_words_set, normalize_tf=None, idf_dict=index.idf,ranking_type="binary_ranking")
            if type_of_index == "our":
                weight = 0.3
                posting_list1 = index.posting_lists_iter_filter(q1_words,300000)
                ##  this is where we binary rank
                output1 = binary_rank(posting_list1)
                if len(output1.keys()) > 0:
                    normelize_factor = len(q1_words)
                    for key in output1:
                        output1[key] = (output1.get(key, 0) * weight * (1 / normelize_factor)) * 0.8 + (df_page_rank.get(key,0) / 9914)  * 0.2

                output = Counter(dict(output1.most_common(3000)))
            elif type_of_index == "regular":
                posting_list1 = index.posting_lists_iter(q1_words)
                output = binary_rank(posting_list1)

            outputQ.put((output, indextype))

    """"
    :argument inputQ-Q that the main process thorw jobs at each index(query) 
    :argument index_config the configuraion for each index ( word2vec , stemmer , where the location all the files
    :argument outputQ after the process (index) has retrived the doc related to the query this is the main pipe that all the process communicate with 
    :argument stop_words_set - is the stopword set that each process will get
    :return this function does not return since its a process so the process puts the answer in the outputQ
    
    this is for the snowball stemmer 
    """""
    def one_core_snow_title(self, inputQ:Queue,index_config:dict,indextype,outputQ,stop_words_set):
        try:
            index = InvertedIndex.read_index(index_config['path_to_index_base_dir'], index_config['index_name'])
            if 'word2df' in index_config:
                path = index_config['word2df']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as w2df:
                        index.df = pickle.load(w2df)
                else:
                    print(f"######## {indextype} {path} didnt found #######")

            if 'doc2len' in index_config:
                path = index_config['doc2len']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2size:
                        index._doc_to_len = pickle.load(d2size)
                else:
                    print(f"######## {indextype} {path} didnt found #######")
            if 'doc2norm' in index_config:
                path = index_config['doc2norm']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path, 'rb') as d2norm:
                        index._doc2norm = pickle.load(d2norm)
                else:
                    print(f"######## {indextype} {path} didnt found #######")
            if 'page-rank' in index_config:
                path = index_config['page-rank']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    with open(path,'rb') as f:
                        df_page_rank = pickle.load(f)
                else:
                    print(f"######## {indextype} {path} didnt found #######")
            if 'word2vec' in index_config:
                path = index_config['word2vec']
                print(f"{indextype} trying to read {path}")
                if self.check_if_path_exists(path):
                    # creating the word2vec model also the Query
                    expander = QueryExpand(stop_words_set, path)
                    q1 = Query('regex', 'snowball')
                    q2 = Query('regex', 'snowball')

        except Exception:
            print(f"Error while loading the {indextype} text index")
            exit(999)
        while True:
            # waiting to receive message(query)
            message = inputQ.get()
            query = message[0]
            limiter = message[1]
            weight = message[2]
            # query expantion using queryexpantion class
            expanded_word_from_query = list(map(lambda tup: tup[0],expander(query,limiter)))
            q1_words = q1(query=expanded_word_from_query, stopwords_frozen=stop_words_set, normalize_tf=None, idf_dict=index.idf,ranking_type="binary_ranking")
            q2_words = q2(query=query, stopwords_frozen=stop_words_set, normalize_tf=None, idf_dict=index.idf,ranking_type="binary_ranking")

            posting_list1 = index.posting_lists_iter_filter(q1_words,300000)
            posting_list2 = index.posting_lists_iter_filter(q2_words,300000)
            ##  this is where we binary rank
            output1 = binary_rank(posting_list1)
            output2 = binary_rank(posting_list2)
            if len(output1.keys()) > 0:
                normelize_factor = len(q1_words)
                for key in output1:
                    output1[key] = 0.1*((output1.get(key, 0) * weight * (1 / normelize_factor)) * 0.8 + (df_page_rank.get(key,0) / 9914) * 0.2)
            if len(output2.keys()) > 0:
                normelize_factor = len(q2_words)
                for key in output2:
                    output2[key] = 0.9*((output2.get(key, 0) * weight * (1 / normelize_factor)) * 0.8 + (df_page_rank.get(key,0) / 9914) * 0.2)

            output = Counter(dict(output1.most_common(3000))) + Counter(dict(output2.most_common(3000)))


            outputQ.put((output, indextype))


    def __init__(self, engine_config_file) -> None:
        self.input_from_process = Queue()
        """
        this is the objects that the engine class holds 
        :dictionary of document ID maped to title
        : dictionary of document ID maped to page view 
        """
        self._doc2title = {}
        self._doc2page_view = {}

        with open(engine_config_file, "r") as f:
            engine_indices_config = json.load(f)
            regular_index_config = engine_indices_config['regular']

            if 'snowball' in engine_indices_config:
                print(f"trying to create snowball index")
                our_index_config = engine_indices_config['snowball']
        """
         this part sets all the dictionary's that relative to the Engine obj
         each dictionary that is common to all the indices is in the same folder as the engine
        """
        if FULL_POWER == 'TRUE':
            print(" ##### starting to read page view to memory ######")
            with open(engine_indices_config['path_to_page_view'], 'rb') as f:
                print(f"trying to create {engine_indices_config['path_to_page_view']}")
                self._doc2page_view = pickle.load(f)
            print(" ##### finish to read page view to memory ###### ")

            print(" ##### starting to read doc 2 title to memory ######")
            with open(engine_indices_config['path_to_doc2title'], 'rb') as f:
                print(f"trying to create {engine_indices_config['path_to_doc2title']}")
                self._doc2title = pickle.load(f)
            print(" ##### finish to read doc 2 title to memory ###### ")


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
                # init_index(self._index_text, reg_idx_txt_cnfg)
                self._index_text.warm_up()
            except Exception:
                print("######Erorr while loading the small power index :((((((########")

        else:
            # creating Queue to comunicate with each process
            self.text_index_q = Queue()
            self.title_index_q = Queue()
            self.anchor_index_q = Queue()
            self.our_text_index_q = Queue()
            self.our_og_txt_index_q = Queue()
            self.our_title_index_q = Queue()
            # the main pipe that all of the process send data back
            self.input_from_process = Queue()
            # configuration files for each of the indices
            rg_txt_cnfg = regular_index_config['text']
            rg_ttl_cnfg = regular_index_config['title']
            rg_anc_cnfg = regular_index_config['anchor']

            #creating the process for each index
            self.pr_text_index = Process(target=self.multi_process_index, args=(self.text_index_q, rg_txt_cnfg, "regular-text-index",self.input_from_process,self._stopwords))
            self.pr_title_index = Process(target=self.multi_process_index, args=(self.title_index_q, rg_ttl_cnfg, "regular-title-index",self.input_from_process,self._stopwords))
            self.pr_anchor_index = Process(target=self.one_core_regular_anchor, args=(self.anchor_index_q, rg_anc_cnfg, "regular-anchor-index",self.input_from_process,self._stopwords))


            or_txt_cnfg = our_index_config['text']
            or_og_txt_cnfg = our_index_config['text_no_w2v']
            or_title_cnfg = our_index_config['title']
            # creating the process for each index
            self.pr_our_text_index = Process(target=self.multi_process_index, args=(self.our_text_index_q, or_txt_cnfg, "our-text-index",self.input_from_process,self._stopwords))
            self.pr_our_og_text_index = Process(target=self.multi_process_index, args=(self.our_og_txt_index_q, or_og_txt_cnfg, "our-text-index",self.input_from_process,self._stopwords))
            self.pr_our_title_index = Process(target=self.one_core_snow_title, args=(self.our_title_index_q, or_title_cnfg, "our-title-index",self.input_from_process,self._stopwords))
            # starting the process as daemons in the background waiting for jobs
            self.pr_anchor_index.start()
            self.pr_title_index.start()
            self.pr_text_index.start()
            self.pr_our_text_index.start()
            self.pr_our_og_text_index.start()
            self.pr_our_title_index.start()



    def search(self, query):

        self.our_text_index_q.put((query, "bm25", "our-index", 3, "on", 0.1))  # -> word2vec
        self.our_og_txt_index_q.put((query, "bm25", "our-index", 3, "off", 0.8))
        self.title_index_q.put((query, "binary", "regular-index", "our", 0.12))
        self.our_title_index_q.put((query, 3, 0.18))
        self.anchor_index_q.put((query, "our"))

        response1 = self.input_from_process.get()
        response2 = self.input_from_process.get()
        response3 = self.input_from_process.get()
        response4 = self.input_from_process.get()
        response5 = self.input_from_process.get()
        dict_1 = response1[0] + response2[0] + response3[0] + response4[0] + response5[0]

        return self.get_top_n_with_title(dict_1, 100)

    def search_body(self, query):
        # sending the text index message that contains the query he needs to process and which similarity function to use
        self.text_index_q.put((query,"cosine","regular-index"))
        # waiting for answer from the index
        result_1 = self.input_from_process.get()
        result = result_1[0]
        return self.get_top_n_with_title(result, 100)

    def search_title(self, query):
        # sending the title index message that contains the query he needs to process and which similarity function to use
        self.title_index_q.put((query,"binary","regular-index","regular"))
        # waiting for answer from the index
        result_1 = self.input_from_process.get()
        result = result_1[0]
        return self.get_top_n_with_title(result, 100)

    def search_anchor(self, query):
        # sending the anchor index message that contains the query he needs to process and which similarity function to use
        self.anchor_index_q.put((query,"regular"))
        # waiting for answer from the index
        result_1 = self.input_from_process.get()
        result = result_1[0]
        return self.get_top_n_with_title(result, 100)

    def get_pagerank(self, lst_of_pages):
        # using the title index that holds the page rank dictionary to give us answers
        self.title_index_q.put((lst_of_pages, "page-rank", "regular-index", "regular"))
        result_1 = self.input_from_process.get()
        result = result_1[0]
        return result

    def get_pageview(self, lst_of_pages):
        # retrive the pages page view with the dictionary
        return [self._doc2page_view[doc_id] for doc_id in lst_of_pages]

    def get_top_n(self, sim_dict, N=3):
        return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]

    def get_top_n_with_title(self, sim_dict, N=3):
        # returns top n docs also retriving their title
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

    """
    See Also return wheter the path exists or not 
    """
    def check_if_path_exists(self, path):
        return os.path.exists(path)

