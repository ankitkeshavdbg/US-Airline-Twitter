# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 19:13:36 2018

@author: Ayush Pandey
"""

import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np

#Read Input csv file
df = pd.read_csv('C:\\Users\\ayush\\Desktop\\US Airline Sentiment\\Tweets.csv', sep = ',')

#shuffle thee training examples in case sentiments are grouped together to remove bias during training
np.random.seed(37)
df = df.reindex(np.random.permutation(df.index))

#drop unnecessary columns
df = df[['text','airline_sentiment']]

#plot count of each label using seaborn
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

sns.factorplot(x = "airline_sentiment", data = df, kind = "count", size = 6)

#df['text'] #df.text does the same

#class to analyze various attributes of the text
import re
import string
import emoji
from sklearn.base import BaseEstimator, TransformerMixin

class TextCounts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, text):
        return len(re.findall(pattern, text))
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x))
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
        count_capital = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
        count_excl_ques_mark = X.apply(lambda x: self.count_regex(r'\!|\?', x))
        count_urls = X.apply(lambda x: self.count_regex(r'http?\S+', x))
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
                                                                                                            
        
        df = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags
                           , 'count_capital': count_capital
                           , 'count_excl_ques_mark': count_excl_ques_mark
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                          })
        
        return df    
        
        
tc = TextCounts()
df_eda = tc.fit_transform(df.text)      

import matplotlib.pyplot as plt
def show_dist(df, col):
    print('Descriptive stats for {}'.format(col))
    print('-'*(len(col)+22))
    print(df.groupby('airline_sentiment')[col].describe())
    bins = np.arange(df[col].min(), df[col].max() + 1)
    g = sns.FacetGrid(df, col='airline_sentiment', size=5, hue='airline_sentiment', palette="PuBuGn_d")
    g = g.map(sns.distplot, col, kde=False, norm_hist=True, bins=bins)
    plt.show()

 # Add airline_sentiment to df_eda
#df_eda['airline_sentiment'] = df.airline_sentiment
#show_dist(df_eda, 'count_words')


#Clean The Text
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class CleanText(BaseEstimator, TransformerMixin):
    
    def remove_mentions(self, text):
        return re.sub(r'@\w+','',text)
    
    def remove_urls(self, text):
        return re.sub(r'https?\S+','',text)
    
    def emoji_oneword(self, text):
        return text.replace('_','')
    
    def remove_punct(self, text):
        # Make translation table
        return re.sub(r'[^\w\s]','',text)
    
    def remove_digits(self, text):
        return re.sub(r'\d+','',text)
    
    def to_lower(self, text):
        return text.lower()
    
    def remove_stopwords(self, text):
        stop_list = stopwords.words('english')
        whitelist = ["n't","not","no"]
        
        words = text.split()
        clean_words = [word for word in words if (word not in stop_list or word in whitelist) and len(word) > 1]
        
        return " ".join(clean_words)

    def stemming(self, text):
        porter = PorterStemmer()
        words = text.split()
        stemmed = [porter.stem(word) for word in words]
        return " ".join(stemmed)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punct).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        
        return clean_X
    
ct = CleanText()
sr_clean = ct.fit_transform(df.text)
sr_clean.sample(5)
sr_clean


print('{} texts have no words after cleaning'.format(sr_clean[sr_clean==''].count()))
sr_clean.loc[sr_clean == ''] = '[no_text]'

from sklearn.feature_extraction.text import CountVectorizer
import collections

cv = CountVectorizer()
bow = cv.fit_transform(sr_clean)
cv.get_feature_names()
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])

df_model = sr_clean.tolist()

class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, **transform_params):
        return X[self.cols]

    def fit(self, X, y=None, **fit_params):
        return self
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(df_model).toarray()
y = df.airline_sentiment

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=37)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predicting
y_pred = classifier.predict(X_test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
    
