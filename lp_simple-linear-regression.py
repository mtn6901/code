import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("D:\DATASET\P14-Part2-Regression\P14-Part2-Regression\Section 6 - Simple Linear Regression\Python\Salary_Data.csv")

X = dataset.iloc[: , 0].values
X = X.reshape(-1,1) 
y = dataset.iloc[: , -1].values
y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
# print(regressor.coef_)

y_pred = regressor.predict(X_test)
a = np.concatenate((X_test, y_test, y_pred), axis = 1)
print(a)

# plt.scatter(X_train, y_train, color = "red")
# plt.plot(X_train, regressor.predict(X_train), color = "green")
# plt.xlabel("years of experience")
# plt.ylabel("salary")
# plt.show()

x0 = np.linspace(1,11,100)
y0 = regressor.coef_*x0

# plt.plot(X, y, 'ro')
# plt.plot(x0,y0)
# plt.axis([0,12,0,130000])
# plt.xlabel('years of experience')
# plt.ylabel('salary') 
# plt.show()