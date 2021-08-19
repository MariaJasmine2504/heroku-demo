# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:04:39 2021

@author: 91861
"""

import numpy as np
import pandas as pd
df = pd.read_csv('C:/Users/91861/Documents/Data Mining- Supervised L/Decision tree/Loan Delinquent Dataset.csv')
for feature in df.columns:
    if df[feature].dtype == 'object':
        df[feature] = pd.Categorical(df[feature]).codes
df=df.drop(['ID','Sdelinquent'],axis=1)
X = df.drop(['delinquent'],axis=1)
y = df.pop('delinquent')
from sklearn.model_selection import train_test_split
X_train, X_test,train_labels_y, test_labels_y = train_test_split(X,y,test_size = .30, random_state=1)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X_train,train_labels_y)
import pickle
pickle.dump(LR,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

