
# Clustering Algorithm to Find Relationships in Complex Dataset
This project aims to find relationships within a complex dataset using a clustering algorithm. The project involves data visualization, normalization, clustering, and determining the optimal number of clusters using the elbow curve method.

# "steps of project" 
Dataset
Dependencies
Implementation
Loading the Dataset
Data Visualization
Data Normalization
Clustering Algorithm
Optimal Number of Clusters
Results
Conclusion

# Dataset
The dataset used in this project is stored in a CSV file named studentclusters.csv. The dataset includes features related to students' marks and study hours.

# Dependencies
The project requires the following Python libraries:
pandas
scikit-learn
matplotlib

# Implementation
# Loading the Dataset
The dataset is loaded using the read_csv function from the pandas library.

# import pandas as pd
dataset = pd.read_csv('studentclusters.csv')
x = dataset.copy()

# Data Visualization
The data is visualized using a scatter plot to understand the distribution of marks and study hours.

x.plot.scatter(x='marks', y='shours')

# Data Normalization
The data is normalized using the minmax_scale function from scikit-learn.

# from sklearn.preprocessing import minmax_scale
X_scaled = minmax_scale(x)

# Clustering Algorithm
The KMeans clustering algorithm is used to fit the normalized data and create clusters.

# from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=1234)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Visualize the clusters
labels = pd.DataFrame(labels)
df = pd.concat([x, labels], axis=1)
df = df.rename(columns={0: 'labels'})
df.plot.scatter(x='marks', y='shours', c='labels', colormap='Set1')

![Screenshot 2024-07-18 175523](https://github.com/user-attachments/assets/e557d92b-1681-460a-ae14-cc116b8e0337)


# Optimal Number of Clusters
The elbow curve method is used to determine the optimal number of clusters by plotting the inertia for different cluster counts.
inertia = []
for i in range(2, 16):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(2, 16), inertia, marker='o')
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('Squared Sum (Inertia)')
plt.show()

![Screenshot 2024-07-18 174557](https://github.com/user-attachments/assets/812284ca-63e9-4c8c-8105-4d07362944ec)


# Results
The clusters are visualized using a scatter plot with different colors representing different clusters.
The elbow curve helps determine the optimal number of clusters by showing the point where the inertia starts to decrease less sharply.

# Conclusion
This project demonstrates the use of the KMeans clustering algorithm to find relationships within a complex dataset. The elbow curve method is effective in determining the optimal number of clusters. Future work can include exploring different clustering algorithms and additional features to improve clustering accuracy.
























