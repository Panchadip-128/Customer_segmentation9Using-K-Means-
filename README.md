# Customer_segmentation_Using-K-Means-


This project applies K-Means clustering to segment customers based on their annual income and spending score. It includes data loading, analysis, determination of optimal clusters, model training, visualization, and a README for setup and usage instructions.

# Customer Segmentation using K-Means Clustering

This project demonstrates the use of K-Means clustering to segment customers based on their annual income and spending score.

## Table of Contents:
----------------------

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Collection & Analysis](#data-collection--analysis)
- [Choosing the Number of Clusters](#choosing-the-number-of-clusters)
- [Training the K-Means Clustering Model](#training-the-k-means-clustering-model)
- [Visualizing the Clusters](#visualizing-the-clusters)
- [Usage](#usage)

## Introduction:
-----------------

Customer segmentation is the practice of dividing a customer base into groups of individuals that have similar characteristics relevant to marketing. In this project, we use the K-Means clustering algorithm to segment customers of a mall based on their annual income and spending score.

## Dependencies:
-----------------

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Data Collection & Analysis:
--------------------------------

We load the customer data from a CSV file, explore its structure, and check for missing values.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('/content/Mall_Customers.csv')

# first 5 rows in the dataframe
customer_data.head()

# finding the number of rows and columns
customer_data.shape

# getting some information about the dataset
customer_data.info()

# checking for missing values
customer_data.isnull().sum()


Choosing the Number of Clusters
We use the elbow method to determine the optimum number of clusters. The elbow point is the number of clusters where the WCSS (Within Clusters Sum of Squares) starts to decrease more slowly.

# selecting the Annual Income and Spending Score columns
X = customer_data.iloc[:, [3, 4]].values

# finding WCSS value for different number of clusters
wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

# plot an elbow graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


From the elbow graph, we determine the optimum number of clusters to be 5.

Training the K-Means Clustering Model
We train the K-Means model with the optimum number of clusters.
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)
print(Y)


Visualizing the Clusters:
------------------------------
We visualize the clusters and their centroids.

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

Usage:
-------
Clone the repository.
Ensure you have the required dependencies installed.
Run the Jupyter notebook or Python script containing the above code.
Load your data into the customer_data DataFrame.
Follow the steps to visualize the clusters.
