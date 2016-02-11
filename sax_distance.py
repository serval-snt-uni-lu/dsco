from os.path import join
import pandas as pd
from prepare_datasets import get_folders
import numpy as np
from saxpy import SAX


def classify(X, y, x, alpha=3):
    sax = SAX(len(x), alpha, 1e-6)
    best, label = np.float('inf'), None
    for i in range(len(X)):
        score = sax.compare_strings(X[i], x)
        if score < best:
            best = score
            label = y[i]
    return label


def knn_classification_sax(args):
    folder, alpha = args
    k=1
    traindf = pd.DataFrame.from_csv(join(folder, 'train', 'saxified_%s.csv' % alpha), index_col=False)
    X = traindf.iloc[:, 1].values.tolist()
    y = traindf.iloc[:, 0].values.tolist()

    testdf = pd.DataFrame.from_csv(join(folder, 'test', 'saxified_%s.csv' % alpha), index_col=False)
    X1 = testdf.iloc[:, 1].values.tolist()
    labelled = testdf.iloc[:, 0].values.tolist()
    predicted = []
    for x in X1:
        predicted.append(classify(X, y, x, alpha=alpha))
    rdf = pd.DataFrame({'label': labelled, 'predicted': predicted})
    rdf.to_csv(join(folder, 'resax_%s_%snn_prediction.csv' % (alpha, k)), index=False)

    return


from multiprocessing import Pool


if __name__ == '__main__':
    params = []
    folders = get_folders('NewlyAddedDatasets/')
    for folder in folders:
        for alpha in range(3, 21):
            params.append((folder, alpha))
    pool = Pool(16)
    pool.map(knn_classification_sax, params)
