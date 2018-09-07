# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np


# In[2]:
#Read the dataset in csv
tweets = pd.read_csv('Tweets.csv')


# In[3]:
#show the first 10 lines
tweets.head(10)


# In[3]:
#Filter for the category
is_positive = tweets['airline_sentiment'].str.contains("positive")
is_negative = tweets['airline_sentiment'].str.contains("negative")  
is_neutral = tweets['airline_sentiment'].str.contains("neutral")
# In[4]:

positive_tweets = tweets[is_positive]
positive_tweets.shape


# In[5]:

negative_tweets = tweets[is_negative]
negative_tweets.shape


# In[6]:

neutral_tweets = tweets[is_neutral]
neutral_tweets.shape

worst_airline = negative_tweets[['airline','airline_sentiment_confidence','negativereason']]
worst_airline


# In[37]:
# Create the rank for the worst airline
cnt_worst_airline = worst_airline.groupby('airline', as_index=False).count()
cnt_worst_airline.sort_values('negativereason', ascending=False)


# In[7]:
# Create the rank for the best airline
best_airline = positive_tweets[['airline','airline_sentiment_confidence']]
cnt_best_airline = best_airline.groupby('airline', as_index=False).count()
cnt_best_airline.sort_values('airline_sentiment_confidence', ascending=False)


# In[11]:
# Create the rank for negative reason
motivation = negative_tweets[['airline','negativereason']]
cnt_bad_flight_motivation = motivation.groupby('negativereason', as_index=False).count()
cnt_bad_flight_motivation.sort_values('negativereason', ascending=False)

import nltk


# In[8]:

nltk.download("punkt")


# In[9]:

nltk.download("stopwords")


# In[13]:

import string
string.punctuation


# Set the useless words:
useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
# In[14]:

def build_bag_of_words_features_filtered(words):
    return {word:1 for word in words if not word in useless_words}

tokenized_negative_tweets = []
for text in negative_tweets['text']:
        tokenized_negative_tweets.append(nltk.word_tokenize(text))
        #negative_words.extend(nltk.word_tokenize(text)) 
        
tokenized_negative_tweets