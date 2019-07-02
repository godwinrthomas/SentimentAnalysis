# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:22:48 2019

@author: Mariam Abbas, Godwin Richard Thomas, Murtuza Mohammed
"""


import pandas as pd
import sys

 
        

data = pd.read_csv('twitter_j2018_full_alpha.csv' , quotechar='|')

data = data.dropna()

import ast #library to convert string to list





#PAPER SUBJECTS


list_of_subjects=[]
#Fetching all subjects from the col paper_subject 
for sub in data['paper_subjects']:
    sub_list = ast.literal_eval(sub) #converts string to list
    sub_index0 = sub_list[0] #taking first subjects from multiple list of subjects
    list_of_subjects.append(sub_index0) #appending all the values of subjects in the list directly




#PUBLISHER SUBJECT
    
list_pub_sub =[]
for pub in data['paper_publisher_subjects']:
    pub_sub_list = ast.literal_eval(pub) #list of dictionaries
    dictionary_1= pub_sub_list[0] # we get the first dictionart in the list
   # type(dictionary_1)
    pub_subject = dictionary_1['name'] 
    list_pub_sub.append(pub_subject)

    
    
#SCOPUS SUBJECTS
    
list_of_scopus_subjects=[]
for scopus_sub in data['paper_scopus_subjects']:
    scopus_sub_list = ast.literal_eval(scopus_sub) #converts string to list
    scopus_sub_index0 = scopus_sub_list[0] #taking first subjects from multiple list of subjects
    list_of_scopus_subjects.append(scopus_sub_index0)   
  

       
#ADDING NEW  COLUMNS TO EXISTING DF
existing_data = data
existing_data =existing_data.assign(subjects=list_of_subjects) 
existing_data=existing_data.assign(Publisher_subjects=list_pub_sub)
existing_data=existing_data.assign(Scopus_Subjects=list_of_scopus_subjects)

#DELETING COLUMNS
del existing_data['paper_subjects']
del existing_data['paper_publisher_subjects']
del existing_data['paper_scopus_subjects']






#COUNTING '#' HASHTAGS 
list_count_Hashtags=[]
for tweet in existing_data['selected_quotes']:
    hash_count = tweet.count("#")
    list_count_Hashtags.append(hash_count)
    
existing_data=existing_data.assign(Count_HashTags = list_count_Hashtags)




#LENGTH OF ABSTRACT- to check sentiments if post is too long or short

list_abstract_len=[]
for abstract in existing_data['paper_abstract']:
    abstract_length= len(abstract)
    list_abstract_len.append(abstract_length)
    
existing_data= existing_data.assign(Abstract_Length = list_abstract_len)








## no of followers of tweet handler (the person who tweets)

'''
start for followers count
'''



'''

type(ast.literal_eval(existing_data['twitter_author_followers'][28]))

type(existing_data['twitter_author_followers'][28])
type(existing_data['twitter_author_followers'][27])



if 'nan' in existing_data['twitter_author_followers'][28]:
    existing_data['twitter_author_followers'][28] = existing_data['twitter_author_followers'][28].replace('nan','0')
    print('j')
    print(type(xx))
print(existing_data['twitter_author_followers'][28])
print(existing_data['twitter_author_followers'][27])


'''


follst=[]
row=0
for followers in existing_data['twitter_author_followers']:
    if 'nan' in followers:
        follst.append(followers.replace('nan',str(0)))
    else:
        follst.append(followers)
    row+=1



existing_data= existing_data.assign(FollowList = follst)



import ast
list_of_followers=[]
for followers in existing_data['FollowList']:
    list_of_followers_row= ast.literal_eval(followers)
    sum=0
    length= len(list_of_followers_row)
    for val in list_of_followers_row:
        sum=sum+val
    avg=sum/length
    list_of_followers.append(int(avg))


existing_data= existing_data.assign(followers_count = list_of_followers)
del existing_data['twitter_author_followers']


del existing_data['FollowList']


'''
end for followers count
'''










#NUMBER OF AUTHOR COUNTS 
import json
import requests
def add_author_count(alt_id):
   try:
       response = requests.get("https://api.altmetric.com/v1/id/" + str(alt_id))
       return len(dict(json.loads(response.content))['authors'])
   except:
       return ''    


rownum=0    
list_authors_count=[]
for altId in existing_data['altmetric_id']:
    number_authors=add_author_count(altId)
    list_authors_count.append(number_authors)
    rownum+=1
    print(rownum)
   # sys.exit()
  
existing_data=existing_data.assign(author_count = list_authors_count) 




#######
##### save existing_data to a new CSV file---> will be the data to work on
#####
existing_data.to_csv('final_data_hurray.csv',index= False)


datadata= pd.read_csv('final_data_hurray.csv')

datadata=datadata.dropna()

