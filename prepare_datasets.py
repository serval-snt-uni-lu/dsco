
# coding: utf-8

# In[ ]:

from scipy.io import loadmat
import pandas as pd
from saxpy import SAX
from os import listdir
from os.path import isfile, join
import os


def saxify_and_export(df, csvf, alphabet=5):
    nrows, ncols = df.shape
    sample_size = ncols - 1
    if sample_size > 100:
        sample_size = 100
    sax = SAX(sample_size, alphabet, 1e-6)
    cols = ['label', 'sax']
    nv = []
    for i in range(nrows):
        values = df.iloc[i, 1:].values.tolist()
        v = {}
        v['label'] = int(df.iloc[i, 0])

        letters, _ = sax.to_letter_rep(values)
        v['sax'] = letters
        nv.append(v)
    return pd.DataFrame(nv, columns=cols).to_csv(csvf, index=False)


def get_folders(path):
    return [join(path, f) for f in listdir(path) if not isfile(join(path, f))]

def get_files(path, ft='.mat'):
    return [join(path, f) for f in listdir(path) if f.endswith(ft) and isfile(join(path, f))]


def saxify_folder(folder, alphabet=(3, 21)):
    files = get_files(folder, ft='.mat')
    for f in files:
        print f
        mat = loadmat(f)
        testdir = join(folder, 'test')
        traindir = join(folder, 'train')
        if not os.path.exists(testdir):
            os.makedirs(testdir)
        if not os.path.exists(traindir):
            os.makedirs(traindir)
        for k in mat.keys():
            if 'test' in k.lower():
                df = pd.DataFrame(mat[k])
                df.to_csv(join(folder, 'test.csv'), index=False)
                for i in range(alphabet[0], alphabet[1]):
                    saxify_and_export(df, join(testdir, 'saxified_%s.csv' % i), alphabet=i)
            if 'train' in k.lower():
                df = pd.DataFrame(mat[k])
                df.to_csv(join(folder, 'train.csv'), index=False)
                for i in range(alphabet[0], alphabet[1]):
                    saxify_and_export(df, join(traindir, 'saxified_%s.csv' % i), alphabet=i)
    return


def saxify_all(path, alphabet=(3, 21)):
    for folder in get_folders(path):
        saxify_folder(folder, alphabet)
    return


if __name__ == '__main__':
    saxify_all('NewlyAddedDatasets')

