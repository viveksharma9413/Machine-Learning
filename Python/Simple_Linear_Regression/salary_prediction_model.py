import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data preprocessing step

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=2/3,random_state=0)

#Applying Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()              #Machine
regressor.fit(X_train,y_train)              #Making the machine to learn