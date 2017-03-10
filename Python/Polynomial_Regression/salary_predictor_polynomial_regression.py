import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
# Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X,y)

# visualization of Linear Regression
plt.scatter(X,y,color='cyan')
plt.plot(X_grid,linear_reg.predict(X_grid),color='red')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Job Levels')
plt.ylabel('Salary')
plt.show()
plt.savefig('Linear Regression smooth graph.png');


# Fitting polynomial regression to the dataset
"""Converting X feature matrix to polynomial feature matrix"""
from sklearn.preprocessing import PolynomialFeatures

for deg in range(2,5):
    poly_reg =PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(X_poly,y)

    #visualization of Polynomial Regression
    plt.scatter(X,y,color='cyan')
    plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='red')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Job Levels')
    plt.ylabel('Salary')
    fig = plt.figure(1)
    fig.canvas.set_window_title('Polynomial Regression smooth graph with degree '+str(deg))
    plt.show()
    plt.savefig('Polynomial Regression smooth graph with degree '+str(deg)+'.png');

