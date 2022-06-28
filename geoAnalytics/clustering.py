from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from numpy import unique


import psycopg2
import numpy as np
import pandas as pd

def kMeans(dataframe, k, max_iter=100):
    """
    K-Means clustering algorithm
    :param dataframe: data to be clustered
    :param k: number of clusters
    :param max_iter: maximum number of iterations
    :return: list of clusters
    """
    # initialize k-means algorithm
    data = dataframe.drop(['x', 'y'], axis=1)
    data = data.to_numpy()
    kmeans = KMeans(n_clusters=k, max_iter=max_iter).fit(data)
    # return the clusters
    #combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=kmeans.labels_)

    return labels, kmeans.cluster_centers_

def kMeansPP(dataframe, k, max_iter=300):
    """
    K-Means++ clustering algorithm
    :param dataframe: data to be clustered
    :param k: number of clusters
    :param max_iter: maximum number of iterations
    :return: list of clusters
    """
    # initialize k-means++ algorithm
    data = dataframe.drop(['x', 'y'], axis=1)
    data = data.to_numpy()
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, init='k-means++').fit(data)
    # return the clusters
    #combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=kmeans.labels_)

    return labels, kmeans.cluster_centers_

def DBScan(dataframe, ep, min_sample):
    # initialize k-means++ algorithm
    data = dataframe.drop(['x', 'y'], axis=1)
    data = data.to_numpy()
    dbs = DBSCAN(eps=ep, min_samples=min_sample).fit(data)
    # return the clusters
    #combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=dbs.labels_)

    return labels

def affinityPropagation(dataframe, damping_factor, max_iter=300, convergence_iter= 15,affinity = 'euclidean',random_state= None,preference=None):
    """
     affinity propagation clustering algorithm
    :param dataframe: data to be clustered
    :param damping factor :In the range [0.5, 1.0) -o avoid numerical oscillations when updating values
    :param max_iter: maximum number of iterations
    :param convergence_iter : default = 15
    :param affinity : {‘euclidean’, ‘precomputed’}
    :param random_state : int, RandomState instance or None, default=None
    :param preference : default = None :points with larger values of preferences are more likely to be chosen as exemplars
    :return: list of clusters
    """
    # initialize affinity propagation algorithm
    data = dataframe.drop(['x', 'y'], axis=1)
    data = data.to_numpy()
    X = StandardScaler().fit_transform(data)
    affinityProp = AffinityPropagation(damping=float(damping_factor), max_iter=int(max_iter), convergence_iter=int(convergence_iter),affinity=affinity, preference=preference, random_state=random_state).fit(X)
    # return the clusters
    #combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=affinityProp.labels_)

    return labels, affinityProp.cluster_centers_


def BIRCH(dataframe, threshold, branch_factor=50, compute_labels=True, n_clusters=3):
    """
     BIRCH - (Balanced Iterative Reducing and Clustering using Hierarchies) involves constructing a tree structure from which cluster centroids are extracted
    :param dataframe: data to be clustered
    :param threshold - (default = 0.5) The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold.
    :param branching_factor - (default = 50) Maximum number of CF subclusters in each node.
    :param n_clusters : n_clustersint, instance of sklearn.cluster model, default=3
    :param compute_labels : True . Whether or not to compute labels for each fit
    :return: list of clusters
    """
    # initialize affinity propagation algorithm
    data = dataframe.drop(['x', 'y'], axis=1)
    data = np.ascontiguousarray(data.to_numpy())

    birch = Birch(n_clusters=int(n_clusters), threshold=float(threshold), branching_factor=int(branch_factor),
                  compute_labels=bool(branch_factor)).fit(data)

    # return the clusters
    # combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=birch.labels_)

    return labels, birch.subcluster_centers_


def agglomerativeClustering(dataframe, n_clusters=2):
    """
    Agglomerative Clustering - Recursively merges pair of clusters of sample data; uses linkage distance
    :param dataframe: data to be clustered
    :param n_clusters : The number of clusters to find.
    :return: list of clusters
    """
    # initialize affinity propagation algorithm
    data = dataframe.drop(['x', 'y'], axis=1)
    data = data.to_numpy()
    # X = StandardScaler().fit_transform(data)

    agglomerativeClustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data)

    # return the clusters
    # combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=agglomerativeClustering.labels_)

    return labels


def meanShift(dataframe, bandwidth=None, max_iters=300):
    """
    Mean shift clustering aims to discover “blobs” in a smooth density of samples.
    :param dataframe: data to be clustered
    :param bandwidth : float, default=None
    :param max_iters : maximum number of iterations, default=300
    :return: list of clusters
    """
    # initialize affinity propagation algorithm
    data = dataframe.drop(['x', 'y'], axis=1)
    data = data.to_numpy()
    # X = StandardScaler().fit_transform(data)

    meanShift = MeanShift(max_iter=max_iters).fit(data)
    # combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=meanShift.labels_)

    return labels, meanShift.cluster_centers_


def opticsClustering(dataframe, min_samples=5, eps=None):
    """
    OPTICS :closely related to DBSCAN, finds core sample of high density and expands clusters from them
    :param dataframe: data to be clustered
    :param min_samples : number of samples in a neighborhood for a point to be considered as a core point.
    :param eps : maximum distance between two samples for one to be considered as in the neighborhood of the other
    :return: list of clusters
    """
    # initialize affinity propagation algorithm
    data = np.ascontiguousarray(np.array(dataframe.drop(['x', 'y'], axis=1)))

    OPTICS_Clustering = OPTICS(min_samples=min_samples, eps=eps).fit(data)
    # combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=OPTICS_Clustering.labels_)

    return labels


def spectralClustering(dataframe, n_clusters=8, assign_labels='discretize'):
    """
     Spectral Clustering - useful when the structure of the individual clusters is highly non-convex
    :param dataframe: data to be clustered
    :param n_clusters : The dimension of the projection subspace
    :param assign_labels : {‘kmeans’, ‘discretize’, ‘cluster_qr’}, default=’kmeans’
    :return: list of cluster labels
    """
    # initialize affinity propagation algorithm
    data = np.ascontiguousarray(np.array(dataframe.drop(['x', 'y'], axis=1)))

    spectralClustering = SpectralClustering(assign_labels=assign_labels, n_clusters=n_clusters, random_state=0).fit(
        data)
    # combine x ,y and labels as dataframe
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=spectralClustering.labels_)

    return labels


def gaussianMixture(dataframe, n_components=1, max_iters=100, covariance_type="full", init_params='kmeans',
                    random_state=0):
    """
     To estimate the parameters of a Gaussian mixture distribution
    :param dataframe: data to be clustered
    :param max_iter : int, default=100
    :param random_state : int, RandomState instance or None, default=None
    :param covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
    :param init_params : {‘kmeans’, ‘k-means++’, ‘random’, ‘random_from_data’}
    :return: clusters,weights(weight of each mixture components, means (mean of each component)),
    """
    # initialize affinity propagation algorithm
    data = np.array(dataframe.drop(['x', 'y'], axis=1))
    gaussianMixture = GaussianMixture(n_components=n_components, max_iter=max_iters, covariance_type=covariance_type,
                                      init_params=init_params, random_state=random_state).fit(data)
    # assign each data point to a cluster
    gaussianResult = gaussianMixture.predict(data)
    label = dataframe[['x', 'y']]
    labels = label.assign(labels=gaussianResult)

    return labels, gaussianMixture.weights_, gaussianMixture.means_


