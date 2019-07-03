# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:23:45 2019

@author: Mariam Abbas, Godwin Richard Thomas, Murtuza Mohammed
"""


from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import random
?
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common import exceptions
?
?
from time import sleep
?
import pandas as pd
df = pd.read_csv('D:/emotionaldata/final_data_hurray.csv')
df.dropna()
?
row_count =df.shape[0]
myList = []
myListAlt=[]
myResult = []
for i in range(0,1500):
    myList.insert(i,df['paper_title'].iloc[i])
    myListAlt.insert(i,df['altmetric_id'].iloc[i])
#print(myList[100])
    
?
chrome = webdriver.Chrome('D:/emotionaldata/chromedriver')
?
ua = UserAgent() # From here we generate a random user agent
proxies = [] # Will contain proxies [ip, port]
?
?
# Retrieve latest proxies
def getProx():
    
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', ua.random)
    proxies_doc = urlopen(proxies_req).read().decode('utf8')
    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')
         # Save proxies in the array
    global proxies
    proxies=[]
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append({
        'ip':   row.find_all('td')[0].string,
        'port': row.find_all('td')[1].string
      })
    print(len(proxies))
    print(proxies)
    
?
?
# Main function
def main():
    # Retrieve latest proxies
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', ua.random)
    proxies_doc = urlopen(proxies_req).read().decode('utf8')
    
    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')
    
    # Save proxies in the array
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append({
                'ip':   row.find_all('td')[0].string,
                'port': row.find_all('td')[1].string
                })
        print(len(proxies))
        print(proxies)
?
?
  # Choose a random proxy
    proxy_index = random_proxy()
    proxy = proxies[proxy_index] 
?
    # Rotating proxy pool
    for n in range(0, 10000):
        try:
            
            req = Request('http://icanhazip.com')
            req.set_proxy(proxy['ip'] + ':' + proxy['port'], 'http')
            sleep(0.5)
            chrome.get("https://scholar.google.com/")
            x = chrome.find_element_by_id('gs_hdr_tsi')
            x.send_keys(myList[n])
            sleep(0.5)
            x.send_keys(Keys.ENTER)
            sleep(0.5)
        
            y = chrome.find_element_by_xpath('//*[@id="gs_res_ccl_mid"]/div[1]/div[2]/div[3]/a[3]')
            z = y.text
            print(str(n) + ':' + z + ':' + myList[n])
            myResult.insert(n,z)
            
        except NoSuchElementException:
            print('NoSuch')
            myResult.insert(n,"n/a")
            pass
        except exceptions.StaleElementReferenceException:
            print('Stale')
            myResult.insert(n,"n/a")
            pass
            
            
        
    
        # Every 10 requests, generate a new proxy
        if n % 5 == 0:
            getProx()
            proxy_index = random_proxy()
            proxy = proxies[proxy_index]
            
    
        # Make the call
        try:
            my_ip = urlopen(req).read().decode('utf8')        
            print('#' + str(n) + ': ' + my_ip)
                
          
        except:  
            # If error, delete this proxy and find another one
            #global proxies
            del proxies[proxy_index]
            print('proxy deleted')
            proxy_index = random_proxy()
            proxy = proxies[proxy_index]
?
# Retrieve a random index proxy (we need the index to delete it if not working)
def random_proxy():
  return random.randint(0, len(proxies) - 1)
?
if _name_ == '_main_':
  main()
  
Final_Result = []
for i in myResult:
    if i == 'n/a':
        Final_Result.append(0)
    elif 'Cited' in i:
        temp = i.split()
        Final_Result.append(int(temp[2]))
    else:
        Final_Result.append(0)
final_df = pd.DataFrame()
?

final_df = final_df.assign(altmetric_id = myListAlt[1000:1100], title = myList[1000:1100], total_citations = Final_Result)
?

final_df.to_csv('D:/emotionaldata/sample11.csv')