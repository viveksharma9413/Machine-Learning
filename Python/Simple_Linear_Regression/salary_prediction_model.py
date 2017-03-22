import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data preprocessing step

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=2/3,random_state=0)

#Applying Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()              #Machine
regressor.fit(X_train,y_train)              #Making the machine to learn

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualizing the training data
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience')
plt.xlabel('Salary (in $)')
plt.show()

#visualizing the testing data
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience')
plt.xlabel('Salary (in $)')
plt.show()