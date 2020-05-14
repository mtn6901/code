# import numpy as np
#
# import pandas as pd
# #importing the Data Set
# dataset = pd.read_csv('/home/nguyen/Documents/Data.csv')
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, 3].values
#
#
# #Handling the missing data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
#
#
# #Encoding categorical data
# from sklearn.preprocessing import LabelEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#
#
# #Train and test
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
#
# #Feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.fit_transform(X_test)
#
#


# import matplotlib.pyplot as mp
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
#
# dataset = pd.read_csv('/home/nguyen/Documents/datasets/studentscores.csv')
# X = dataset.iloc[:, : 1].values
# Y = dataset.iloc[:,  1].values
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4, random_state=0)
#
# regressor = LinearRegression().fit(X_train, Y_train)
#
# Y_pred = regressor.predict(X_test)
#
# mp.scatter(X_train, Y_train, color='red')
# mp.plot(X_train, regressor.predict(X_train), color='blue')
#
# # mp.scatter(X_test , Y_test, color = 'red')
# # mp.plot(X_test , regressor.predict(X_test), color ='blue')


# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('/home/nguyen/Documents/datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding Categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
X = np.array(pd.concat([ pd.DataFrame(X[:, :3]), pd.get_dummies(X[:, 3])], axis = 1))
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categories= [3] )
# X = onehotencoder.fit_transform(X).toarray()
# ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],remainder='passthrough')
# X = ct.fit_transform(X)

print(X)

# # Avoiding Dummy Variable Trap
# X = X[:, 1:]
#
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
# # Step 2: Fitting Multiple Linear Regression to the Training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
#
# # Step 3: Predicting the Test set results
# y_pred = regressor.predict(X_test)

