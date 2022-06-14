from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
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