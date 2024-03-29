import nltk
import string 
import re
import numpy as np
import pandas as pd 
import pickle 

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style="white")
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.corpus import stopwords
from sklearn.feature_extraction import _stop_words 

from collections import Counter 
from wordcloud import WordCloud 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.offline as py 
#py.init_nodebook_mode(connection=True)
import plotly.graph_objs as go
import plotly.tools as tls 
#%matplotlib inline 

import bokeh.plotting as bp 
from bokeh.models import HoverTool, BoxSelectTool 
from bokeh.models import ColumnDataSource 
from bokeh.plotting import figure, show, output_notebook 

import warnings 
warnings.filterwarnings('ignore')
import logging 
logging.getLogger("Ida").setLevel(logging.WARNING)

# read in the Mercari, Japan's biggest community-powered shopping app, training data set 
# the task is to give pricing suggestions to sellers 
# the goal is to build an algorithm that can automatically suggests the right product prices. 
train = pd.read_csv('train.tsv', sep='\t')
print(train.dtypes)
print(train.head())
print(train.price.describe())

# plot the price distribution and log price distribution
plt.subplot(1,2,1)
(train['price']).plot.hist(bins=50, figsize=(20,10), edgecolor='white',range=[0,250])
plt.xlabel('price+', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Price Distribution - Training Set', fontsize=17)

plt.subplot(1,2,2)
np.log(train['price']).plot.hist(bins=50, figsize=(20,10), edgecolor='white',range=[0,250])
plt.xlabel('log(price+)', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Log(Price) Distribution - Training Set', fontsize=17)
plt.show()
# most prices are in the lower range, 20ish, convert to log will make the price distribution more normal
train.shipping.value_counts()/len(train)

prc_shipBySeller = train.loc[train.shipping==1, 'price']
prc_shipByBuyer = train.loc[train.shipping==0, 'price']

# plots the price distribution for whether the shipping cost is covered by the seller 
# use the different alpha parameter, the transparency degree to uncover the two distributions 
fig, ax = plt.subplots(figsize=(20,10))
ax.hist(np.log(prc_shipBySeller+1), color='#8CB4E1', alpha=1.0, bins=50, label='Price when Seller pays Shipping')
ax.hist(np.log(prc_shipByBuyer+1), color='#007D00', alpha=0.7, bins=50, label='Price when Buyer pays Shipping')
ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
plt.legend()
plt.xlabel('log(price+1)', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.title('Price Distribution by Shipping Type', fontsize=17)
plt.show()
# the price distribution of seller cover shipping is higher than not cover the shipping 

# how many unique values in the category column
print("There are %d unique values in the category column"% train['category_name'].nunique())

# top 5 raw categories 
print(train['category_name'].value_counts()[:5])
# missing categories 
print("There are %d items that do not have a label."%train['category_name'].isnull().sum())
# divide more clearly, divide the category strings by the / line 
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

# create new columns for sub categories using the lambda function 
train['general_cat'], train['subcat_1'], train['subcat_2'] = \
zip(*train['category_name'].apply(lambda x: split_cat(x)))
print(train.head())
print("There are %d unique first sub-categories." % train['subcat_1'].nunique())
print("There are %d unique second sub-categories." % train['subcat_2'].nunique())
# 114 and 871 first and second sub categories 
x = train['general_cat'].value_counts().index.values.astype('str')
y = train['general_cat'].value_counts().values 
pct = ['%.2f'%(v*100)+"%" for v in (y/len(train))]

# use plotly to plot the interactive graph
# the main category of spending by distribution 
tracel = go.Bar(x=x, y=y, text=pct)
layout = dict(title='Number of Items by Main Category', yaxis = dict(title='Count'), 
              xaxis= dict(title='Category'))
fig = dict(data=[tracel], layout=layout)
py.iplot(fig)

# subcat_1 distribution 
x = train['subcat_1'].value_counts().index.values.astype('str')[:15]
y = train['subcat_1'].value_counts().values 
pct = ['%.2f'%(v*100)+"%" for v in (y/len(train))][:15]

# use plotly to plot the interactive graph
# the sub 1 category of spending by distribution, top 15 ones
trace1 = go.Bar(x=x, y=y, text=pct,
                marker = dict(
                    color = y, colorscale='Portland', showscale=True, 
                    reversescale=False
                ))
layout = dict(title='Number of Items by Sub Category (Top 15)', 
              yaxis = dict(title='Count'), 
              xaxis= dict(title='SubCategory'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)

# box plot for category and price 
general_cats = train['general_cat'].unique()
x = [train.loc[train['general_cat']==cat, 'price'] for cat in general_cats]
data = [go.Box(x=np.log(x[i]+1), name=general_cats[i]) for i in range(len(general_cats))]
layout = dict(title="Price Distribution by General Category",
              yaxis = dict(title='Frequency'),
              xaxis = dict(title='Category'))
fig = dict(data= data, layout = layout)
py.iplot(fig)

# brand name analysis 
print("There are %d unique brand names in the training dataset." % train['brand_name'].nunique())
# Brand distributions for top 10 brands 
x=train['brand_name'].value_counts().index.values.astype('str')[:10]
y = train['brand_name'].value_counts().values[:10]
trace1 = go.Bar(x=x, y=y, 
                marker=dict(
                    color = y, colorscale='Portland',showscale=True,
                    reversescale=False
                ))
layout = dict(title='Top 10 Brand by Number of Items',
              yaxis= dict(title='Brand Name'),
              xaxis = dict(title='Count'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)

# the product description is non-strcutural. First do the data processing, remove all puctures, stopwords, small lengths
def wordCount(text):
    try:
        text = text.lower()
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        # tokenize 
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words 
        words = [w for w in txt.split(" ") \
                 if not w in _stop_words.ENGLISH_STOP_WORDS and len(w) > 3]
        return len(words)
    except:
        return 0 
# add a column of word counts to both the training and test set 
train['desc_len'] = train['item_description'].apply(lambda x: wordCount(x))
print(train.head())

df = train.groupby('desc_len')['price'].mean().reset_index()
# does length of name has anything to do with price 
trace1 = go.Scatter(
    x = df['desc_len'],
    y = np.log(df['price']+1),
    mode = 'lines+markers',
    name = 'lines+markers'
)
layout = dict(title='Average Log(Price) by Description Length',
              yaxis = dict(title='Average Log(Price)'),
              xaxis = dict(title='Description Length')
)
fig = dict(data=[trace1], layout = layout)
py.iplot(fig)
# price moves up as the description goes up initially, after 27 length then price decreases 
# and the price flutuates a lot when the length is long 
print(train.item_description.isnull().sum())
# remove these four missing values
train = train[pd.notnull(train['item_description'])]
# create a dictionary of words for each category
tokenize = nltk.data.load('tokenizers/punkt/english.pickle')

cat_desc = dict()
for cat in general_cats:
    text = " ".join(train.loc[train['general_cat']==cat, 'item_description'].values)
    cat_desc[cat] = tokenize.tokenize(text)
# flat list of all words combined 
flat_list = [item for sublist in list(cat_desc.values()) for item in sublist]
allWordsCount = Counter(flat_list)
all_top10 = allWordsCount.most_common(20)
x = [w[0] for w in all_top10]
y = [w[1] for w in all_top10]
trace1 = go.Bar(x=x, y=y, text=pct)
layout = dict(title='Word Frequency',
              yaxis = dict(title='Count'),
              xaxis = dict(title='Word'))
fig = dict(data=[trace1], layout=layout)
py.iplot(fig)
# word cloud 
stop = set(stopwords.words('english'))
def tokenize(text):
    try:
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text) # remove punctuation
        # tokenize 
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words 
        tokens = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w) >=3]
        return filtered_tokens
    except TypeError as e: print(text, e)

train['tokens'] = train['item_description'].map(tokenize)
train.reset_index(drop=True, inplace=True)

for description, tokens in zip(train['item_description'].head(),
                               train['tokens'].head()):
    print('description:', description)
    print('tokens:', tokens)
    print()

# build dictionary with key = category and values as all the descriptions related 
cat_desc = dict()
for cat in general_cats:
    text = " ".join(train.loc[train['general_cat']==cat, 'item_description'].values)
    cat_desc[cat] = tokenize(text)

# find the most common words for the top 4 categories 
women100 = Counter(cat_desc['Women']).most_common(100)
beauty100 = Counter(cat_desc['Beauty']).most_common(100)
kids100 = Counter(cat_desc['Kids']).most_common(100)
electronics100 = Counter(cat_desc['Electronics']).most_common(100)

def generate_wordcloud(tup):
    wordcloud = WordCloud(background_color='white', 
                          max_words = 50, max_font_size=40,
                          random_state=42).generate(str(tup))
    return wordcloud
# words cloud plots 
fig, axes = plt.subplots(2,2,figsize=(30,15))
ax = axes[0,0]
ax.imshow(generate_wordcloud(women100), interpolation='bilinear')
ax.axis('off')
ax.set_title("Women Top 100", fontsize=30)

ax = axes[0,1]
ax.imshow(generate_wordcloud(beauty100), interpolation='bilinear')
ax.axis('off')
ax.set_title("Beauty Top 100", fontsize=30)

ax = axes[0,0]
ax.imshow(generate_wordcloud(kids100), interpolation='bilinear')
ax.axis('off')
ax.set_title("Kids Top 100", fontsize=30)

ax = axes[0,0]
ax.imshow(generate_wordcloud(electronics100), interpolation='bilinear')
ax.axis('off')
ax.set_title("Eletronics Top 100", fontsize=30)


# tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 10, max_features=180000, tokenizer= tokenize, ngram_range=(1,2))
all_desc = np.append(train['item_description'].values, test['item_description'].values)
vz = vectorizer.fit_transform(list(all_desc))
print(vz.shape)

