import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#dibaties dataset

data = pd.read_csv("/Users/onkarkottawar/Documents/LP-3/diabetes.csv")

df = data.copy()

print(df.head())

print(df.tail())

print(df.info())

print(df.describe())

print(df.isnull().sum())

#replacing 0 values with NaN for specific columns

column_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for column in column_to_replace:
    df[column] = df[column].replace(0, np.nan)
    df[column].fillna(df[column].mean(), inplace=True)

print(df.head())

print(df.isnull().sum())

x = df.iloc[:, :8]
y = df['Outcome']

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

plt.figure(figsize=(10, 6))
sns.boxplot(df)
plt.title("outlier detection using boxplot")
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Number of outliers in each column:\n", outliers)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

print(knn)

knn_pred = knn.predict(x_test)

cm = metrics.confusion_matrix(y_test, knn_pred)
accuracy = metrics.accuracy_score(y_test, knn_pred)
precision = metrics.precision_score(y_test, knn_pred)
recall = metrics.recall_score(y_test, knn_pred)
f1 = metrics.f1_score(y_test, knn_pred)
error_rate = 1 - accuracy

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Error Rate:", error_rate)

accuracy_list = []

for k in (3, 5, 7,):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, knn_pred)
    accuracy_list.append(accuracy)
    print(f"Accuracy for k={k}: {accuracy}")

plt.plot([3, 5, 7], accuracy_list, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.grid()
plt.show()