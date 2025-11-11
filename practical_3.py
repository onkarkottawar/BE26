import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#salse data

data = pd.read_csv("/Users/onkarkottawar/Documents/LP-3/sales_data_sample.csv", encoding="latin1")

df = data.copy()

print(df.head())

print(df.tail())

print(df.info())

print(df.describe())

print(df.isnull().sum())

to_drop = ["ADDRESSLINE1", "ADDRESSLINE2", "STATE", "POSTALCODE", "PHONE"]

df = df.drop(to_drop, axis=1)

print(df.isnull().sum())

print(df.dtypes)

df_numeric = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 6))
sns.boxenplot(data=df_numeric)
plt.title('Boxen Plot of Numeric Features')
plt.show()

Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
print("Number of outliers in each numeric column:\n", outliers)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_numeric)
print("data normalized using StandardScaler")    

df_normalized = pd.DataFrame(x_scaled, columns=df_numeric.columns)

print("sample of normalized data:")
print(df_normalized.head())

print("mean of normalized data:",df_normalized.mean())
print("std of normalized data:",df_normalized.std())

intertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    intertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, intertia, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(x_scaled)
    silhouette_avg = silhouette_score(x_scaled, cluster_labels)
    print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(x_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_normalized.iloc[:, 0], y=df_normalized.iloc[:, 1], hue=labels, palette='Set1')
plt.title('K-Means Clustering Results')
plt.show()  