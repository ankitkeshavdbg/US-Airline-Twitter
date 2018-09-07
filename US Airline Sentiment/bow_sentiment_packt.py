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
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punct).apply(self.remove_digits).apply(self.to_lower)#.apply(self.remove_stopwords).apply(self.stemming)
        
        return clean_X
    
ct = CleanText()
sr_clean = ct.fit_transform(df.text)
sr_clean.sample(5)

df.loc[df['airline_sentiment']=='negative', 'airline_sentiment']=0
df.loc[df['airline_sentiment']=='positive','airline_sentiment']=1
df.loc[df['airline_sentiment']=='neutral','airline_sentiment'] = 1
X = sr_clean
y = df.airline_sentiment

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 37)

print('Count Accuracy : ')
cv1 = CountVectorizer()

X_train_cv  = cv1.fit_transform(X_train)
X_train.iloc[0]
y_train.iloc[0]
len(X_train_cv.toarray())

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

y_train = y_train.astype('int64')
mnb.fit(X_train_cv, y_train)

X_test_cv = cv1.transform(X_test)
pred = mnb.predict(X_test_cv)
pred
actual = np.array(y_test)
count = 0
for i in range (len(pred)):
    if pred[i]==actual[i]:
        count = count + 1

acc=count/len(pred)*100
print(acc)

#TFIDF
print('Tfidf Accuracy : ')
cv1 = TfidfVectorizer()

X_train_cv  = cv1.fit_transform(X_train)
X_train.iloc[0]
y_train.iloc[0]
len(X_train_cv.toarray())

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

y_train = y_train.astype('int64')
y_test = y_test.astype('int64')
mnb.fit(X_train_cv, y_train)

X_test_cv = cv1.transform(X_test)
pred = mnb.predict(X_test_cv)
pred
actual = np.array(y_test)
count = 0
for i in range (len(pred)):
    if pred[i]==actual[i]:
        count = count + 1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)

acc=count/len(pred)*100
acc
#dtm = X_train_cv.transform(text)
#
#pd.DataFrame(dtm.toarray(), columns = cv.get_feature_names())
