# -*- coding: utf-8 -*-
# @Time  : 2023/2/5 14:49
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
from copy import deepcopy
from typing import List

import numpy as np


def cluster(array: np.ndarray, tol=0.0) -> np.ndarray:
    """Cluster the array and return the group numbers of all elements.

    Args:
        array: np.ndarray, 1d array to be grouped. support dtype (float, int, str)
        tol: tolerance to spilt.

    Returns:
        group_label: np.ndarray, group numbers of all elements.

    Examples:

        >>> arr = np.array([2, 4, 4, 3, 1, 4, 2])
        >>> cluster(arr)
        array([1, 3, 3, 2, 0, 3, 1])

        >>> arr = np.array(["b","s","s","l","a","s","b"])
        >>> cluster(arr)
        array([1, 3, 3, 2, 0, 3, 1])
    """

    array = array.ravel()
    array_index = np.argsort(array)
    array_sort = array[array_index]
    array_index_iv = np.argsort(array_index)

    if tol == 0.0:
        _, split_index = np.unique(array_sort, return_index=True)
        change = np.zeros_like(array_index, dtype=np.int64)
        change[split_index] = 1
    else:
        array_sort2 = np.append(np.nan, array_sort[:-1])
        if not isinstance(array_sort, (np.float, int)):
            change = array_sort2 != array_sort
        else:
            try:
                change = np.abs(array_sort2 - array_sort) < tol
            except BaseException:
                raise NotImplementedError(f"Not support dtype {array.dtype}.")
        change = change.astype(np.int64)
    num = np.cumsum(change) - 1
    clu = num[array_index_iv]
    return clu


def cluster_split(array: np.ndarray, tol=0.0) -> List[np.ndarray]:
    """Cluster the array and return the split groups.

    Args:
        array: np.ndarray, 1d array to be grouped. support dtype (float, int, str)
        tol: tolerance to spilt.

    Returns:
        groups: list of np.ndarray, groups.

    Examples:

        >>> arr = np.array([2, 4, 4, 3, 1, 4, 2])
        >>> cluster_split(arr)
        [array([4]), array([0, 6]), array([3]), array([1, 2, 5])]

        >>> arr = np.array(["b","s","s","l","a","s","b"])
        >>> cluster_split(arr)
        [array([4]), array([0, 6]), array([3]), array([1, 2, 5])]
    """

    array = array.ravel()
    array_index = np.argsort(array)
    array_sort = array[array_index]

    array_sort2 = np.append(np.nan, array_sort[:-1])

    if tol == 0.0:
        _, split_index = np.unique(array_sort, return_index=True)
        change = split_index[1:]
    else:
        if not isinstance(array_sort, (np.float, int)):
            change = array_sort2 != array_sort
        else:
            try:
                change = np.abs(array_sort2 - array_sort) < tol
            except BaseException:
                raise NotImplementedError(f"Not support dtype {array.dtype}.")

        change = np.where(change)[0][1:]

    gps = np.split(array_index, indices_or_sections=change)

    return gps


def coarse_and_spilt_array(array: np.ndarray, tol: float = 0.5, method: str = None,
                           n_cluster: int = 3, reverse: bool = False) -> np.ndarray:
    """
    Split 1D ndarray by distance or group.

    Args:
        array: (np.ndarray) with shape (n,).
        tol: (float) tolerance distance for spilt.
        method:(str) default None. others: "agg", "k_means", "cluster", "k_means_user".
        n_cluster: (int) number of cluster.
        reverse:(bool), reverse the label.

    Returns:
        labels: (np.ndarray) with shape (n,).

    """
    if method in ["agg", "k_means"]:
        if method == "agg":
            from sklearn.cluster import AgglomerativeClustering
            ac = AgglomerativeClustering(n_clusters=None, distance_threshold=tol, compute_distances=True)
        else:
            from sklearn.cluster import KMeans
            ac = KMeans(n_clusters=n_cluster)

        ac.fit(array.reshape(-1, 1))
        labels_ = ac.labels_
        labels_max = np.max(labels_)
        labels = deepcopy(labels_)
        dis = np.array([np.mean(array[labels_ == i]) for i in range(labels_max + 1)])
        dis_index = np.argsort(dis)
        for i in range(labels_max + 1):
            labels[labels_ == i] = dis_index[i]
    else:
        # use tol directly
        labels = cluster(array, tol=tol)

    if reverse:
        labels = max(labels) - labels
    return labels
