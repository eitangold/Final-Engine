from tokenize import tokenize
import re
from tokenize import tokenize

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


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
    def get_tokenizer(tokenizer:str):
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
            return PorterStemmer()
        elif stemmer == 'snowball':
            return SnowballStemmer('english')
        elif stemmer == 'regular':
            return None

        