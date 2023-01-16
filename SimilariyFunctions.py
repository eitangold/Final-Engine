############ this is the part where are all the similarity functions ############
import math
from collections import Counter

"""
The cosine_similarity function is used to rank the documents in the collection based on their similarity 
to a given query, using the cosine similarity measure.
"""
def cosine_similarity(query, w_posting_list_itr, idf_d, d_len, d_norm):
    score_dict = {}

    for w, post_list in w_posting_list_itr:
        q_tfidf = query.tf[w]
        idf_word = idf_d[w]

        q_tfidf_mul_idf_wordForDoc = q_tfidf * idf_word
        for d_id, d_tf in post_list:
            score_dict[d_id] = score_dict.get(d_id, 0) + (q_tfidf_mul_idf_wordForDoc * d_tf)

    q_norm = query.norm
    for d_id, score in score_dict.items():
        score_dict[d_id] = score_dict[d_id] / (q_norm * d_norm[d_id] * d_len[d_id])
    return score_dict

"""
The binary_rank function is used to rank the documents in the collection based on their presence of the query terms.
"""
def binary_rank(w_posting_list_itr):
    score_dict = Counter()

    for w, post_list in w_posting_list_itr:
        temp_dict = Counter(dict(post_list))
        score_dict += temp_dict

    return score_dict

"""
The BM25_score function is used to rank the documents in the collection based on their relevance to a given query, using the BM25 ranking algorithm.
Returns
A dictionary where the keys are the document IDs and the values are the relevance scores for each document
"""
def BM25_score(w_posting_list_itr, d_idf, d_len, avgdl):
    # b is between [0-1] k_query and k_doc are between [0-infinity]
    score_dict = Counter()
    B_dict = {}
    b = 0.8
    k_doc = 0.3
    for w, post_list in w_posting_list_itr:
        w_idf = d_idf[w]
        for d_id, d_tf in post_list:
            if d_id not in B_dict:
                B_dict[d_id] = 1 - b + (b * (d_len[d_id] / avgdl))

            score_dict[d_id] = score_dict.get(d_id, 0) + \
                               ((((k_doc + 1) * d_tf) / (B_dict[d_id] * k_doc + d_tf)) * w_idf)

    return score_dict
