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


X = data1.iloc[:, :-1].values
y = data1.iloc[:, 8].values


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

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculate MAE, MSE, RMSE
import numpy as np
from sklearn import metrics
print("MAE ",metrics.mean_absolute_error(y_test, y_pred))
print("MSE ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import r2_score
print("R-Square ",r2_score(y_test, y_pred))




# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculate MAE, MSE, RMSE
import numpy as np
from sklearn import metrics
print("MAE ",metrics.mean_absolute_error(y_test, y_pred))
print("MSE ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import r2_score
print("R-Square ",r2_score(y_test, y_pred))




# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculate MAE, MSE, RMSE
import numpy as np
from sklearn import metrics
print("MAE ",metrics.mean_absolute_error(y_test, y_pred))
print("MSE ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import r2_score
print("R-Square ",r2_score(y_test, y_pred))



# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculate MAE, MSE, RMSE
import numpy as np
from sklearn import metrics
print("MAE ",metrics.mean_absolute_error(y_test, y_pred))
print("MSE ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE ",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import r2_score
print("R-Square ",r2_score(y_test, y_pred))

