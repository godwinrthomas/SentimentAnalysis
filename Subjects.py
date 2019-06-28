# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:18:43 2019

@author: Mariam Abbas, Godwin Richard Thomas, Murtuza Mohammed
"""

import pandas as pd

data = pd.read_csv('twitter_j2018_full_alpha.csv', quotechar='|')

data = data.dropna()


dictionary_subjects={}
#PAPER_SUBJECTS
import ast
list_of_sub=[]
for sub in data['paper_subjects']:
    list_of_sub= ast.literal_eval(sub)
    for ss in list_of_sub:
        if ss not in dictionary_subjects:
            dictionary_subjects[ss]=1
        else:
            dictionary_subjects[ss]+=1
    
 
     
#SCOPUS SUBJECTS

dictionary_scopus_subjects={}
#PAPER_SUBJECTS
import ast
list_of_sub=[]
for sub in data['paper_scopus_subjects']:
    list_of_sub= ast.literal_eval(sub)
    for ss in list_of_sub:
        if ss not in dictionary_scopus_subjects:
            dictionary_scopus_subjects[ss]=1
        else:
            dictionary_scopus_subjects[ss]+=1
        
        
    
#PUBLISHER SUBJECTS

import ast
dictionary_pub_subject = {}
list_dics=[]
for pub in data['paper_publisher_subjects']:
    list_dics = ast.literal_eval(pub)
    for subs in list_dics:
        subs = str(subs)
        each_sub = ast.literal_eval(subs)
        
        if each_sub['name'] not in dictionary_pub_subject:
            dictionary_pub_subject[each_sub['name']]=1
            
        else:
            dictionary_pub_subject[each_sub['name']]+=1
            
            
        
import csv
with open('subject.csv', 'w') as f:
    f.write("Subject,Count\n")
    for key in dictionary_subjects.keys():
        f.write("%s,%s\n"%(key,dictionary_subjects[key]))
        
import csv
with open('scopus_subject.csv', 'w') as f:
    f.write("Scopus Subject,Count\n")
    for key in dictionary_scopus_subjects.keys():
        f.write("%s,%s\n"%(key,dictionary_scopus_subjects[key]))
        
import csv
with open('Publisher_subject.csv', 'w') as f:
    f.write("Publisher Subject,Count\n")
    for key in dictionary_pub_subject.keys():
        f.write("%s,%s\n"%(key,dictionary_pub_subject[key]))
        
        
        
        
            
