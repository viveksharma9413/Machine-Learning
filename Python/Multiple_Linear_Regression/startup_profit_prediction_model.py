import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt

#data preprocessing step
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,-1] = labelencoder.fit_transform(X[:,-1])
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
"""This step is not required as most of the libraries 
take care of this step by themselves"""
X = X[:,1:]

#splitting the dataset into traing and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)