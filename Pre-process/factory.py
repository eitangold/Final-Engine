import nltk
from tokenize import tokenize
from abc import ABC, abstractmethod
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
class TokenizerInterface:
    def __init__(self,tokenizerfunc) -> None:
        self._tokenizer = tokenizerfunc
    def tokenize(self,sentence:str)->object:
        return self._tokenizer(sentence)


#the purpose of this factory is to return object of all  the configuration from the configuration file
class FactoryIndex:
    def __init__(self) -> None:
        pass
    @staticmethod
    def get_tokenizer(tokenizer:str)->object:
        if tokenizer == 'nltk':
            pass
        elif tokenizer == 'python':
            return TokenizerInterface(tokenize)
        elif tokenizer == 'regex':
            REGEX_TOK = re.compile(r"[\#\@\w](['\-]?\w){2,24}", re.UNICODE)
            return TokenizerInterface(REGEX_TOK.finditer)
    @staticmethod
    def get_stemmer(stemmer:str)->object:
        if stemmer == 'porter':
            logger.critical("@@@@ this is creation of porter from factory @@@@ ")
            return PorterStemmer()
        elif stemmer == 'snowball':
            logger.critical("@@@@ this is creation of snowball from factory @@@@ ")
            return SnowballStemmer('english')
