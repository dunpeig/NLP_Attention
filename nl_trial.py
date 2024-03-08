import nltk
from nltk.tokenize import word_tokenize 
from nltk.text import Text

# tokenize 
input_str = "today's weather is good, very windy and sunny, we have no classes in the afternoon. We will play basketball in the court."
tokens = word_tokenize(input_str)
tokens = [word.lower() for word in tokens]
print(tokens[:5])

# Text 
t = Text(tokens)
print(t.count('good'))
print(t.index('good'))
#t.plot(8)

# stopwords 
from nltk.corpus import stopwords 
print(stopwords.readme().replace('\n', ' '))
print(stopwords.fileids())
print(stopwords.raw('english').replace('\n', ' '))

test_words = [word.lower() for word in tokens]
test_words_set = set(test_words)
print(test_words_set.intersection(set(stopwords.words('english'))))

# filter out the stopwords 
filtered = [w for w in test_words if(w not in stopwords.words('english'))]
print(filtered)


