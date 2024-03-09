# data clearning 
import re 
from nltk.corpus import stopwords 
import nltk
from nltk.tokenize import word_tokenize 
from nltk.text import Text

s = ' RT @ Amila #Test\nTom\'s newly listed Co    &amp: Mary\'s unlisted        Group to supply tech for nltk. \nh $TSLA $AAPL https:// t.co/x34afsfQsh'

cache_english_stopwords = stopwords.words('english')

def text_clean(text):
    print('rawdata:', text, '\n')

    # remove html tags 
    text_no_special_entities = re.sub(r'\&\w*;|#\w*|@\w*', '', text)
    print('without special tags:', text_no_special_entities, '\n')

    # remove value tags 
    text_no_tickers = re.sub(r'\$\w*', '', text_no_special_entities)
    print('no value tags:', text_no_tickers, '\n')

    # remobe hyperlinks 
    text_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', text_no_tickers)
    print('no hyperlinks:', text_no_hyperlinks, '\n')

    # remove special terms 
    text_no_small_words = re.sub(r'\b\w{1,2}\b', '', text_no_hyperlinks)
    print('no small words:', text_no_small_words, '\n')

    # remove extra spaces 
    text_no_whitespace = re.sub(r'\s\s+', ' ', text_no_small_words)
    text_no_whitespace = text_no_whitespace.lstrip(' ')

    print('without whitespace:', text_no_whitespace, '\n')

    tokens = word_tokenize(text_no_whitespace)
    print('result:', tokens, '\n')

text_clean(s)






