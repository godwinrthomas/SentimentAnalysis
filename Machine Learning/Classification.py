# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:07:02 2019

@author: Mariam Abbas, Godwin Richard Thomas, Murtuza Mohammed
"""

import pandas as pd

data= pd.read_csv('final_sentiment_score.csv')

data= data.dropna()

data1 = data[['Scopus_Subjects','title_sentiment_score','abstract_sentiment_score','Count_HashTags',
              'twitter_rt_count', 'followers_count', 'author_count',
              'Abstract_Length','tweet_sentiment_scores']]


import numpy as np
X = data1.iloc[:, :-1].values
y = data1.iloc[:, 8].values

yy=[]
 
for i in range(len(y)):
    if y[i]>= -1 and y[i]< -0.2:
        yy.append(-1)
    elif y[i]>= -0.2 and y[i]<0.2:
        yy.append(0)
    else:
        yy.append(1)
        
newyy= np.array(yy)


y=newyy
    


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

'''
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

XX=X

##Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""
sc_y = StandardScaler()
y_train = y_train.reshape(-1,1)
y_train = sc_y.fit_transform(y_train)
"""




"""
ALL ML MODELS
"""


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=20, min_samples_split=1500)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics
sklearn.metrics.accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [{'criterion': ["gini"], 'max_depth': [20,50,100], 'min_samples_split': [1500,5000,10000]},
               {'criterion': ["entropy"], 'max_depth': [20,50,100], 'min_samples_split': [1500,5000,10000]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

from sklearn.metrics import classification_report
report =classification_report(y_test, y_pred)


"""----------------------
Accuracy : 0.670399139177511

             precision    recall  f1-score   support

         -1       0.66      0.72      0.69      7092
          0       0.66      0.71      0.68     11991
          1       0.70      0.59      0.64     10656

avg / total       0.67      0.67      0.67     29739

"""


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy',max_depth=20,min_samples_split=300, bootstrap=False, n_estimators=80)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics
sklearn.metrics.accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [{'criterion': ["entropy"], 'max_depth': [10,20,40], 'min_samples_split': [300,600,1000], 'n_estimators':[40,60,80], 'bootstrap':[False]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

from sklearn.metrics import classification_report
report =classification_report(y_test, y_pred)

"""----------------------
accuracy 0.6731564612125491

             precision    recall  f1-score   support

         -1       0.66      0.72      0.69      7092
          0       0.67      0.68      0.68     11991
          1       0.69      0.63      0.66     10656

avg / total       0.67      0.67      0.67     29739


"""


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics
sklearn.metrics.accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()


"""----------------------"""


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics
sklearn.metrics.accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

"""----------------------

accuracy 0.6368068865799119
"""


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics
sklearn.metrics.accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

"""----------------------

accuracy 0.6101079390699082
"""


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics
sklearn.metrics.accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()