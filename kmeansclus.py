import sys

import numpy as np
import pandas
from matplotlib import pyplot as plt
from numpy.linalg import norm
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=10000000, random_state=150):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break


if __name__ == '__main__':
    k = int(input('Enter the value of K: '))
    if k<=0:
        print('Invalid input')
        exit()
    if len(sys.argv)>1:
        file = sys.argv[1]
    else:
        file= 'iris-dataset.csv'
    dataset = pandas.read_csv(file)
    X_std = StandardScaler().fit_transform(dataset)
    km = Kmeans(k)
    km.fit(X_std)
    centroids = km.centroids
    colors=['b','g','r','c','m','y']
    # Plot the clustered data
    for i in range(k):
        plt.scatter(X_std[km.labels == i, 0], X_std[km.labels == i, 12],c=colors[i], label='cluster '+str(i))
    plt.legend()
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()
    '''iris = load_iris()
    df = pandas.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    y = df.iloc[0:150, 4].values
    X = StandardScaler().fit_transform(df.iloc[0:150, [0, 2]].values)
    # plot the actual data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='blue', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='green', marker='o', label='Versicolor')
    plt.scatter(X[100:150, 0], X[100:150, 1],
                color='red', marker='o', label='Virginica')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.legend()
    plt.show()'''

