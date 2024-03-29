from nltk.corpus import stopwords 
import nltk
from nltk.tokenize import word_tokenize 
from nltk.text import Text
import pandas as pd
import spacy 
import collections
# define nlp method
nlp = spacy.load('en_core_web_sm')
# terrorist articles from RAND database
terrorism_articles = pd.read_csv('RAND_Database_of_Worldwide_Terrorism_Incidents.csv',encoding='utf-8', encoding_errors='replace')
terrorism_articles = terrorism_articles['Description']
print(terrorism_articles[:5])

terrorism_articles_nlp = [nlp(art) for art in terrorism_articles]

common_terrorist_group = [
    'taliban',
    'al - qaeda',
    'hamas',
    'fatah',
    'plo',
    'bilad a1 - rafidayn'
]

common_locations = [
    'iraq',
    'baghdad',
    'kirkuk',
    'mosul',
    'afghanistan',
    'kabul',
    'basra',
    'palestine',
    'gaza',
    'israel',
    'istanbul',
    'beirut',
    'pakistan'
]

location_entity_dict = collections.defaultdict(collections.Counter)

# the data is problematic, something wrong processing below
for article in terrorism_articles_nlp:
    article_terrorist_groups = [ent.lemma_ for ent in article.ents if ent.label_ =='PERSON' or ent.label_ == 'ORG']
    article_locations = [ent.lemma_ for ent in article.ents if ent.label_ =='GPE']
    terrorist_common = [ent for ent in article_terrorist_groups if ent in common_terrorist_group]
    locations_common = [ent for ent in article_locations if ent in common_locations]

    for found_entity in terrorist_common:
        for found_location in locations_common:
            location_entity_dict[found_entity][found_location] += 1

print(location_entity_dict)

location_entity_df = pd.DataFrame.from_dict(dict(location_entity_dict))
location_entity_df = location_entity_df.fillna(value=0).astype(int)
print(location_entity_df)

import matplotlib.pyplot as plt 
import seaborn as sns 

plt.figure(figsize=(12,10))
hmap = sns.heatmap(location_entity_df, annot=True, fmt='d', cmap='Y1GnBu', char=False)

plt.title('Global Incidents by Terrorist group')
plt.xticks(rotation=30)
plt.show()


