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
(np.log(train['price'])).plot.hist(bins=50, figsize=(20,10), edgecolor='white')
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
x = train['general_cat'].value_counts
