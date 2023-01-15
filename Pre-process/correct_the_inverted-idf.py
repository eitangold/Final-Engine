import math
import pickle
from collections import Counter
import numpy as np




def create_idf_dict(name_file_df, name_file_doc_len):

    with open(f'{name_file_df}', 'rb') as w2df:
        df_dict = pickle.load(w2df)

    with open(f'{name_file_doc_len}', 'rb') as d2size:
        doc_to_len = pickle.load(d2size)

    idf_dict ={}
    N = len(doc_to_len)
    for word, df in df_dict.items():
        idf_dict[word] = math.log2(N / df)


def doc_to_count(text, id, idf_dict, stemmer=None, idx_type=None):

    if stemmer is not None:
        tokens = [stemmer.stem(token.group()) for token in Tok.tokenize(text.lower()) if
                  token.group() not in all_stopwords]
    else:
        tokens = [token.group() for token in Tok.tokenize(text.lower()) if token.group() not in all_stopwords]

    token_countr = Counter(tokens)
    doc_len = len(token_countr.values())
    tf_idf = [(tf / doc_len) * idf_dict[w] for w, tf in token_countr.items()]
    doc_size = np.linalg.norm(np.array(tf_idf))


    return id, (np.round(doc_size, 5), doc_len)


