import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


scaler = None

# Data reading, cleaning, and splitting
df = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
)

# Drops the columns that aren't related to song attributes
df_new = df.drop(df.columns[:11], axis=1)

# Drops rows which contain missing values
if df_new.isnull().sum().any() > 0:
    df_new = df_new.dropna()



# Random State is a seed to make results reproducible
# Splitting into train, validations, and test set
X_k_means, X_knn = train_test_split(
    df_new, test_size=0.2, random_state=206
)

# Tensors need numeric types so converting booleans to numeric values
boolean_columns = X_k_means.select_dtypes(include="bool").columns
X_k_means[boolean_columns] = X_k_means[boolean_columns].astype(int)

boolean_columns = X_knn.select_dtypes(include="bool").columns
X_knn[boolean_columns] = X_knn[boolean_columns].astype(int)

# Scaling Data (prevents features with different scales being dominant)
scaler = StandardScaler()

# Only fit on the train data since we only have access to the train data
scaler.fit(X_k_means)
joblib.dump(scaler, "scaler.pkl")

# Need to scale all the data in order to preprocess for model since it expects
# the same scaled input it was trained on (so both sets must be scaled)
X_k_means_scaled = scaler.transform(X_k_means)
X_knn_scaled = scaler.transform(X_knn)


kmeans_cluster = KMeans(init = "k-means++", n_clusters=k, n_init=10, random_state=123)
#kmeans_cluster.fit(X_k_means)


# Reduced Fit
pca = PCA(2)
reduced_data = pca.fit_transform(X_k_means)
kmeans_cluster.fit(reduced_data)

# Calculating the centroids
centroids = kmeans_cluster.cluster_centers_
label = kmeans_cluster.fit_predict(reduced_data)
unique_labels = np.unique(label)

# plotting the clusters:
plt.figure(figsize=(8, 8))
for i in unique_labels:
    plt.scatter(reduced_data[label == i, 0],
                reduced_data[label == i, 1],
                label=i)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='k', zorder=10)
plt.legend()
plt.show()

    

    

