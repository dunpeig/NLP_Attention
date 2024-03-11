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

    # remove stop words 
    list_no_stopwords = [i for i in tokens if not i in cache_english_stopwords]
    print("remove stopwords: ", list_no_stopwords, '\n')

    # results after filter 
    text_filtered = ' '.join(list_no_stopwords)
    print("results after filter: ", text_filtered)

text_clean(s)


# spaCy
import spacy 

# English 
nlp = spacy.load('en_core_web_sm')

doc = nlp('Weather is good, very windy and sunny. We have no classes in the afternoon.')
# divide words 
for token in doc:
    print(token)
# divide sentences
for sent in doc.sents:
    print(sent)

for token in doc:
    print(' {}-{}'.format(token,token.pos_))


doc_2 = nlp("I went to Paris where I met my old friend Jack from uni.")

from spacy import displacy
doc = nlp('I went to Paris where I met my old friend Jack from uni.')
displacy.render(doc, style='ent', jupyter=True)


def read_file(file_name):
    with open(file_name, 'r') as file:
        return file.read()
    
# use the book of Pride and Prejudice as an example to find the most frequent names of characters 
text = read_file('pride_and_prejudice.txt')
processed_text = nlp(text)
sentences = [s for s in processed_text.sents]
print(len(sentences))

print(sentences[:5])

from collections import Counter 
def find_person(doc):
    c = Counter()
    for ent in processed_text.ents:
        if ent.label_ == 'PERSON':
            c[ent.lemma_] += 1
    return c.most_common(10)
print(find_person(processed_text))



