# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:37:11 2019

@author: Mariam Abbas, Godwin Richard Thomas, Murtuza Mohammed
"""



import pandas as pd

data = pd.read_csv('tweet_preprocessed_sentiment.csv' )

data = data.dropna()
 #CLEANING THE SELECTED QUOTES
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
title = []
abstract=[]


for val in data['paper_title']:
    review = re.sub('[^a-zA-Z]',' ', val)
    review = review.lower()
    final_review = review.split()
    #list of all words including stop words
    #final_review=[]
    #list od all words excluding stopwords
    final_review2=[]
    for word in final_review :
        if word not in set(stopwords.words('english')):
            final_review2.append(word)
   # ps = PorterStemmer()
    #final_review = [ps.stem(word) for word in final_review if not word in set(stopwords.words('english'))]
    final_review2 = ' '.join(final_review2)
    title.append(final_review2)
    
for val in data['paper_abstract']:
    review = re.sub('[^a-zA-Z]',' ', val)
    review = review.lower()
    final_review = review.split()
    #list of all words including stop words
    #final_review=[]
    #list od all words excluding stopwords
    final_review2=[]
    for word in final_review :
        if word not in set(stopwords.words('english')):
            final_review2.append(word)
   # ps = PorterStemmer()
    #final_review = [ps.stem(word) for word in final_review if not word in set(stopwords.words('english'))]
    final_review2 = ' '.join(final_review2)
    abstract.append(final_review2)
    

existing_data=data
existing_data =existing_data.assign(title_stop_rem=title) 
existing_data =existing_data.assign(abstract_stop_rem=abstract) 



#existing_data.to_csv('tweet_preprocessed2.csv')

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
# first, we import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# next, we initialize VADER so we can use it within our Python script
sid = SentimentIntensityAnalyzer()

title_sentiment_list=[]
abstract_sentiment_list=[]


for message_text in existing_data['title_stop_rem']:
    scores = sid.polarity_scores(message_text)
    compound_score=scores['compound']
    title_sentiment_list.append(compound_score)
    
for message_text in existing_data['abstract_stop_rem']:
    scores = sid.polarity_scores(message_text)
    compound_score=scores['compound']
    abstract_sentiment_list.append(compound_score)
    
    
existing_data2= existing_data
existing_data2=existing_data2.assign(title_sentiment_score=title_sentiment_list)
existing_data2=existing_data2.assign(abstract_sentiment_score=abstract_sentiment_list)
 

existing_data2.to_csv('final_sentiment_score.csv')




