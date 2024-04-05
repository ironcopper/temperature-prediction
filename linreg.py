import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# loading the data
temperature = pd.read_csv("DailyDelhiClimate.csv")
  
temperature.shape       # (rows, columns)

# checking for missing values if any
print(temperature.isnull().sum())

# A boxplot visually displays the distribution of the data and identifies outliers as individual points beyond the whiskers
plt.boxplot(temperature['meantemp'])  # target variable(or dependent variable)

# visualising the data
sns.pairplot(temperature, x_vars=['humidity', 'meanpressure', 'wind_speed'], y_vars='meantemp', height=4, aspect=1, kind='scatter')
plt.show()

# checking the correlation between variables
sns.heatmap(temperature.corr(), cmap="YlGnBu", annot = True)
plt.show()

X = temperature['humidity']     # the feature variable, humidity
y = temperature['meantemp']     # the response variable, meantemp


# splitting data into training and testing sets; keeping 70% in train 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

print("X_train shape:", X_train.shape) 
print("X_test shape:", X_test.shape) 
print("y_train shape:", y_train.shape) 
print("y_test shape:", y_test.shape) 


# reshaping the data
X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

print("X_train shape:", X_train.shape) 
print("X_test shape:", X_test.shape) 
print("y_train shape:", y_train.shape) 
print("y_test shape:", y_test.shape) 


# building the model
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()     # create a linear regression model
model = linreg.fit(X_train, y_train)     # train the model on the train set

print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)


# making predictions on test set
y_pred = model.predict(X_test)

# showing data and fitted line together 
plt.scatter(X_train, y_train)
plt.plot(X_train, model.intercept_ + model.coef_*X_train, 'r')
plt.show()


# displaying the predicitons and test values
df = pd.DataFrame({"Actual value": y_test, "Predicted Value": y_pred})


from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error

mse = mean_squared_error(y_test, y_pred)

print(f"mse: {mse:.3f}")
print(f"rmse: {np.sqrt(mse):.3f}")
print(f"mae: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"r-squared: {r2_score(y_test, y_pred):.3f}")
print(f"mape: %{100*mean_absolute_percentage_error(y_test, y_pred):.3f}")
print(f"max error: {max_error(y_test, y_pred):.3f}")