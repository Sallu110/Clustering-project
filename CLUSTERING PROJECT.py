
# CLUSTERING ALGORITHEM TO FIND RELATION IN COMPLEX DATASET 


import pandas as pd 

dataset = pd.read_csv('studentclusters.csv')
x = dataset.copy()

# visualize the data using pandas 
x.plot.scatter(x = 'marks', y = 'shours')

# normalize the data using standard or min-max 

from sklearn.preprocessing import minmax_scale 

X_scaled = minmax_scale(x) 


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters  = 3 , random_state = 1234)

# fit the data to get clusters 
kmeans.fit(X_scaled)

labels = kmeans.labels_

# visualize the clusters 

labels = pd.DataFrame(labels)
df = pd.concat([x,labels], axis  = 1)

df = df.rename(columns = {0:'labels'})

df.plot.scatter(x = 'marks', y = 'shours', c = 'labels', colormap = 'Set1')

# elbow method for optimum clusters 

inertia = []

for i in range(2,16):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt 

plt.figure()
plt.plot(range(2,16),inertia,marker = 'o')
plt.title('ELBOW CURVE')
plt.xlabel(' NUMBER OF CLUSTER')
plt.ylabel('squared sum(interia)')

# PROJECT END 