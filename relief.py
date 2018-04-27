import random
import numpy as np
import pandas as pd
from  Consts import inf

def relief_nearst_miss(X: np.ndarray, Y: np.ndarray, index: int):
    tag = Y[index]
    value = X[index]
    relevant_locations = np.argwhere(Y!=tag)
    closest_location = min(relevant_locations, key= lambda x: abs(value - X[x[0]]))
    return value - X[closest_location[0]]

def relief_nearst_hit(X: np.ndarray, Y: np.ndarray, index: int):
    tag = Y[index]
    value = X[index]
    relevant_locations = np.argwhere(Y==tag)
    closest_location = min(relevant_locations, key= lambda x: abs(value - X[x[0]]) if index != x[0] else inf)
    return value - X[closest_location[0]]


def relief_alg(X: pd.DataFrame, Y: pd.DataFrame,N: int, tau: float) -> set:
    result = set()

    npY = np.array(Y)
    for feature in X.keys():
        npX = np.array(X[feature])
        w = 0.0
        indexes = random.sample(range(X.shape[0]),N)
        for index in indexes:

            w += relief_nearst_miss(npX, npY, index) ** 2
            w -= relief_nearst_hit(npX, npY, index) ** 2

        if w >= tau:
            result.add(feature)

    return result