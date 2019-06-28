# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:08:04 2019

@author: maria
"""

import pandas as pd
import numpy as np
import sys


import os
os.getcwd()
os.chdir("C:\\Users\\maria\\Desktop\\IRS PROJECT NEW DATA")

data_pos_Date = pd.read_csv('twitter_j2018_full_alpha.csv',quotechar='|')

#dropna not availavle values are discarded
data_pos_Date = data_pos_Date.dropna()

import ast
#YEAR VALUES 
dictionary_year = {}
list_of_dates=[]
for date in data_pos_Date['tweet_post_date']:
    list_of_dates= ast.literal_eval(date)
    for dd in list_of_dates:
        year_value = dd[0:4]
    #print(year_value)
        
        if year_value not in dictionary_year:
            dictionary_year[year_value]=1
        
        else:
            dictionary_year[year_value]+=1
            
del dictionary_year['2018']
        
import csv
with open('year_trend.csv', 'w') as f:
    f.write("Year,Count\n")
    for key in dictionary_year.keys():
        f.write("%s,%s\n"%(key,dictionary_year[key]))