import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def sequential_forward_selection(clf, X: pd.DataFrame, y: pd.DataFrame, k) -> list:
    """
    calculate for each available amount of features the best set.
    like in the tutor, large sets contain the smaller sets.

    :return: a dict indexed by int's, each entry contains a set of the best features selected for this entry.
    """


    X = X.loc[:, X.columns != 'Unnamed: 0']
    base = [feature for feature in X.keys()]
    print(base)
    bestIndexes = dict()
    bestScores = dict()
    X = X.as_matrix()
    y = y.as_matrix().ravel()
    currScore = 0
    for i in range(k):
        bestScore = 0
        for j in range(0, len(base)):
            if j in bestIndexes.values():
                continue
            currIndexes = [bestIndexes[k] for k in range(i)]
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
    indexByOrder = []
    bestFeatures = []
    for k in bestIndexes.keys():
        indexByOrder.append(bestIndexes[k])
        bestFeatures.append(base[bestIndexes[k]])
    # bestFeatures = [base[i] for i in range(len(base)) if i in indexByOrder]
    return indexByOrder, bestFeatures