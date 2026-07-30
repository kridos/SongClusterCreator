import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import umap



# Data reading, cleaning, and splitting
df = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
)


# Drops rows which contain missing values
if df.isnull().sum().any():
    df = df.dropna()



# Random State is a seed to make results reproducible
# Splitting into k means and k nearest neighbors set
X_k_means, X_knn = train_test_split(
    df, test_size=0.2, random_state=206
)

# Drops the columns that aren't related to song attributes
X_k_means_dropped = X_k_means.drop(X_k_means.columns[:11], axis=1)
X_knn_dropped = X_knn.drop(X_knn.columns[:11], axis=1)

# Tensors need numeric types so converting booleans to numeric values
boolean_columns = X_k_means_dropped.select_dtypes(include="bool").columns
X_k_means_dropped[boolean_columns] = X_k_means_dropped[boolean_columns].astype(int)

boolean_columns = X_knn_dropped.select_dtypes(include="bool").columns
X_knn_dropped[boolean_columns] = X_knn_dropped[boolean_columns].astype(int)

# Scaling Data (prevents features with different scales being dominant)
scaler = StandardScaler()

# Only fit on the k means data since that is the data we are training on
scaler.fit(X_k_means_dropped)

# Need to scale all the data in order to preprocess for model since it expects
# the same scaled input it was trained on (so both sets must be scaled)
X_k_means_scaled = scaler.transform(X_k_means_dropped)
X_knn_scaled = scaler.transform(X_knn_dropped)


# List to store inertias at different Ks
# Interia:  sum of squared distances between each data point and its assigned cluster centroid
# indicating how tightly the points are grouped within clusters
inertias = []

# Finds interias from k means from k = 2 to k = 11
for k in range(2, 11):
  # Finds clusters with more strategic initalization (k means ++)
  kmeans_cluster = KMeans(init = "k-means++", n_clusters=k, n_init=10, random_state=123)
  kmeans_cluster.fit(X_k_means_scaled)
  inertias.append(kmeans_cluster.inertia_)

# Displays inertias in a line plot
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(range(2, 11), inertias)
plt.show()

# Input based on elbow in graph and final training on chosen k
final_k = int(input("What is the k value that you chose: "))
kmeans_cluster = KMeans(init = "k-means++", n_clusters=final_k, n_init=10, random_state=123)

# Calculating the labels for each point (which centroid each point is in)
label = kmeans_cluster.fit_predict(X_k_means_scaled)

# Calculates each of the centroids
k_mean_centroids = kmeans_cluster.cluster_centers_

# Labels
unique_labels = np.unique(label)

# Reduces dimensions 
reducer = umap.UMAP(n_neighbors=15, random_state=42)
umap_data = reducer.fit_transform(X_k_means_scaled)

# Calculates centroids based on mean of each cluster
umap_centroids = np.array([umap_data[label == i].mean(axis=0) for i in unique_labels])

# Plotting Clusters and Centroids
fig2, ax2 = plt.subplots(figsize=(8,8))
# plotting the clusters:
for i in unique_labels:
    ax2.scatter(umap_data[label == i, 0],
                umap_data[label == i, 1],
                label=i)
ax2.scatter(umap_centroids[:, 0], umap_centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='k', zorder=10)
ax2.legend()
plt.show()

# Calculates Centroids based on reducing k means centroids 
# Looks better visually on the graph but doesn't represent UMAP as well
reduced_centroids = reducer.transform(k_mean_centroids)

# Plotting Clusters and Centroids
fig3, ax3 = plt.subplots(figsize=(8,8))
# plotting the clusters:
for i in unique_labels:
    ax3.scatter(umap_data[label == i, 0],
                umap_data[label == i, 1],
                label=i)
ax3.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='k', zorder=10)
ax3.legend()
plt.show()


# Assigns label numbers to cluster of data
clusters = {}
for i in unique_labels:
  clusters[i] = X_k_means[label == i]
  
# Prints out two most popular songs in each cluster (can contain duplicates)
for key, value in clusters.items():
  print(f"Category {key}:")
  value_sorted = value.sort_values(by='track_popularity', ascending=False)
  print(value_sorted.iloc[0])
  print(value_sorted.iloc[1])
  print()
  
# Manual Cluster Labels based on Songs
cluster_labels = {
  0: "Mid-energy pop/R&B",
  1: "Acoustic/mellow pop",
  2: "Speechy mid-energy tracks",
  3: "High energy EDM/dance",
  4: "Latin/reggaeton",
  5: "High energy mainstream pop",
  6: "Mixed acoustic/electronic pop"
}

# Intializing and fitting knn classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_k_means_scaled, label)

# Predicting which cluster knn data songs will be in
y_pred = knn.predict(X_knn_scaled)

# Saves output to a file
with open('classification.txt', 'w', encoding='utf-8') as f:
  for i in range(len(y_pred)):
      f.write(f"Classifier Predicted that {X_knn.iloc[i, 1]} by {X_knn.iloc[i, 2]} is {cluster_labels[y_pred[i]]}\n")
    

    

