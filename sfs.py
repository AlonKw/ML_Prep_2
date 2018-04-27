import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def sequential_forward_selection(clf, X: pd.DataFrame, y: pd.DataFrame, k) -> list:
    """
    calculate for each available amount of features the best set.
    like in the tutor, large sets contain the smaller sets.

    :return: a dict indexed by int's, each entry contains a set of the best features selected for this entry.
    """

    base = [feature for feature in X.keys()]
    bestIndexes = dict()
    bestScores = dict()
    print(base)
    X = X.as_matrix()
    y = y.as_matrix().ravel()
    currScore = 0
    for i in range(1, k):
        bestScore = 0
        bestIndex = 1
        for j in range(1, len(base)):
            if j in bestIndexes.values():
                continue
            currIndexes = [x for x in bestIndexes]
            currIndexes.append(j)
            currX = X[:, currIndexes]
            tempScore = metrics.accuracy_score(y, cross_val_predict(clf, currX, y, cv=3))
            if tempScore > bestScore:
                bestScore = tempScore
                bestIndexes[i] = j
                bestScores[i] = bestScore


        bestFeatures = [base[i] for i in range(len(base)) if i in bestIndexes.values()]
        print(bestIndexes)
        print(bestFeatures)
        print(bestScores)
    bestFeatures = [base[i] for i in range(len(base)) if i in bestIndexes.values()]
    return bestIndexes