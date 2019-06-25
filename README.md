# SentimentAnalysis
  
  The main aim of the project was to predict the emotions of all the research articles by using Twitter data via. Altmetrics dataset. The
first phase of the project involved preprocessing the data and the use of Automated Web Scraping for feature selection. The second
phase was to find the sentiment of the tweets with the help of Natural Language Processing (NLP) via. NLTK (Natural Language ToolKit). Finally, Machine Learning
models were used to predict the emotions.
  
  The Altmetrics Dataset was used and a list of 15 features were selected. Using the base features, some features were derived and
then appended into the dataset. A feature called "Total Number of Citations" was scraped automatically using the Selenium module
in Python and added as a feature.

  Pre-processing was done to remove the stop words, hyperlinks and punctuations. The intensity of sentiments were then found using the NLTK. 
The sentiment score for the features Title, Abstract and Actual Tweet were found and the compound score was calculated as their average
and used as the target variable.

  Machine learning models were then applied for the prediction analysis. Both classification and regression algorithms were used and
the matrics for evaulation were accuracy, precision, recall and f1-score. The best performing model was Support Vector Machines (SVM).

  For future work, Artifical Neural Networks (ANN) could be implemented for better performance.
    
   
