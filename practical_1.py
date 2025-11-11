import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#uber dataset

data = pd.read_csv("/Users/onkarkottawar/Documents/PYTHON/uber.csv")

df = data.copy()

print(df.head())

print(df.tail())

print(df.info())

print(df.describe())

print(df.isnull().sum())

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])   

print(df.info())

print(df.describe())

print(df.isnull().sum())

df.select_dtypes(include=[np.number]).corr()

print(df.columns)

df.dropna(inplace=True)

plt.figure(figsize=(10, 6))
plt.boxplot(df["fare_amount"])
plt.title("Boxplot of Fare Amount")
plt.ylabel("Fare Amount")
plt.show()

q_low = df["fare_amount"].quantile(0.01)
q_hi = df["fare_amount"].quantile(0.99)

df_filtered = df[(df["fare_amount"] > q_low) & (df["fare_amount"] < q_hi)]

print(df.isnull().sum())

from sklearn.model_selection import train_test_split

x = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

x['pickup_datetime'] = pd.to_numeric(pd.to_datetime(x['pickup_datetime']))
x = x.loc[:, x.columns.str.contains('^Unnamed')]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.linear_model import LinearRegression

lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)

predict = lrmodel.predict(x_test)

print(predict)

from sklearn.metrics import mean_squared_error, r2_score

lr_rmse = np.sqrt(mean_squared_error(y_test, predict))
lr_r2 = r2_score(y_test, predict)

print("Linear Regression RMSE:", lr_rmse)
print("Linear Regression R2 Score:", lr_r2)

from sklearn.ensemble import RandomForestRegressor
rfmodel = RandomForestRegressor(n_estimators=100, random_state=1)

rfmodel.fit(x_train, y_train)
rf_predict = rfmodel.predict(x_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predict))
rf_r2 = r2_score(y_test, rf_predict)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2 Score:", rf_r2)

