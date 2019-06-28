# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:04:12 2019

@author: maria
"""


import pandas as pd

# quotechar = '|' is for separating the columns
data_pub_Date = pd.read_csv('twitter_j2018_full_alpha.csv',quotechar='|')

#dropna not availavle values are discarded
data_pub_Date = data_pub_Date.dropna()


#YEAR VALUES 
dictionary_year = {}

for date in data_pub_Date['paper_pubdate']:
    year_value = date[0:4]
    #print(year_value)
    
    if year_value not in dictionary_year:
        dictionary_year[year_value]=1
    
    else:
        dictionary_year[year_value]+=1
        
del dictionary_year['2018']
        
import csv
with open('publish_year_trend.csv', 'w') as f:
    f.write("Year,Count\n")
    for key in dictionary_year.keys():
        f.write("%s,%s\n"%(key,dictionary_year[key]))