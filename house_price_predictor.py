import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
#Data Set
boston = load_boston()
#Plotting the Data Set
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target
print(df.head())
print(df.describe())
sns.pairplot(df, vars=['RM', 'LSTAT', 'AGE', 'PRICE'])
plt.show()
#Training Data
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
#Plotting
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
