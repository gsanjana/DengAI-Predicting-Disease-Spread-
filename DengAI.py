# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:14:20 2020

@author: SANJANG
"""

import numpy as np
import pandas as pd
import math 

#Importing Data
csv_train = 'DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv'
csv_test = 'DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv'
csv_train_y = 'DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv'
df_train= pd.read_csv(csv_train,parse_dates=['week_start_date'],header=0,index_col=False,keep_default_na=True)
df_test= pd.read_csv(csv_test,parse_dates=['week_start_date'],header=0,index_col=False,keep_default_na=True)
df_train_y = pd.read_csv(csv_train_y,header=0,index_col=False,keep_default_na=True)

#Replacing NaN values
# Replace using median 
i=0
for (columnName) in df_train.columns.values:
    i=i+1
    if i>4:
        median = df_train[columnName].median()
        df_train[columnName].fillna(median, inplace=True)
        
#print(df_train.isnull().sum())
df_train['day'] = df_train['week_start_date'].dt.day
df_train['month'] = df_train['week_start_date'].dt.month
df_train.drop(columns =['week_start_date'], inplace = True)

i=0
for (columnName) in df_test.columns.values:
    i=i+1
    if i>4:
        median = df_test[columnName].median()
        df_test[columnName].fillna(median, inplace=True)
        
df_test['day'] = df_test['week_start_date'].dt.day
df_test['month'] = df_test['week_start_date'].dt.month
df_test.drop(columns =['week_start_date'], inplace = True)   
        
#Training and Testing
X = df_train.iloc[:, :].values
y = df_train_y.iloc[:, 3].values   
X_test = df_test.iloc[:,:].values      
        
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Onehotfor X_test
X_test_copy = X_test.copy()
labelencoder_X_test = LabelEncoder()
X_test_copy[:, 0] = labelencoder_X_test.fit_transform(X_test_copy[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X_test_copy = onehotencoder.fit_transform(X_test_copy).toarray()

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X_test_copy)
y_pred = y_pred.tolist()
y_pred = [round(x) for x in y_pred]
y_pred = [int(x) for x in y_pred]
dataframe = pd.DataFrame()
dataframe['city'] =  df_test['city']
dataframe['year'] = df_test['year']
dataframe['weekofyear'] =  df_test['weekofyear']
dataframe['total_cases'] = y_pred

dataframe.to_csv('DengAI_Predicting_Disease_Spread_-_Test_Data_Labels.csv', index=False) 