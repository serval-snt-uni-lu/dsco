
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd
import os

from collections import defaultdict

from itertools import islice

from prepare_datasets import get_folders, get_files

from os import listdir
from os.path import isfile, join
import os


max_word_size = 20

def build_words(seq, n=2):
    # See http://stackoverflow.com/a/7636054
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def build_corpus(sax_csv_file, corpus_dir, min_word_size=2, max_word_size=5):
    df = pd.DataFrame.from_csv(sax_csv_file, index_col=False)
    nrows, ncols = df.shape
    dfs = []
    sizes = []
    for label in df['label'].unique():
        ndf = df[df['label'] == label]
        dfs.append(ndf)
        sizes.append(ndf.shape[0])


    normalize_to = max(sizes)*1000.0
    
    interval = min_word_size + 1
    interval = 1
    cdir = corpus_dir
    if not os.path.exists(cdir):
        try:
            os.makedirs(cdir)
        except:
            pass
    for df in dfs:
        # http://stackoverflow.com/a/17171819
        # rows = np.random.choice(df.index.values, sample_size)
        # df = df.ix[rows]

        l = len(df['sax'].iloc[0])
        c = df['label'].iloc[0]

        dic = defaultdict(int)
        dic2 = defaultdict(int)
        for idx, row in df.iterrows():
            s = row['sax']
            for size in range(min_word_size, max_word_size + 1, interval):
                words = [''.join(x) for x in build_words(s, size)]
                for i in range(len(words)):
                    dic[words[i]] += 1
                for j in range(size):
                    for i in range(j, len(words), size):
                        if i == 0:
                            dic2[('<S>', words[i])] += 1
                        elif i - size > 0:
                            dic2[(words[i-size], words[i])] += 1

        with open('%s/bigram_wl_%s_to_%s_class_%s.txt' % (cdir, min_word_size, max_word_size, c), 'w') as f:
            for k in dic2.keys():
                v = int(dic2[k] * normalize_to / df.shape[0])
                f.write('%s %s\t%s\n' % (k[0], k[1], v))
        with open('%s/unigram_wl_%s_to_%s_class_%s.txt' % (cdir, min_word_size, max_word_size, c), 'w') as f:
            for k in dic.keys():
                v = int(dic[k] * normalize_to / df.shape[0])
                f.write('%s\t%s\n' % (k, v))
    return



def build_corpus_for_folder(folder):
    for i in range(3, 21):
        traincsv = join(folder, 'train', 'saxified_%s.csv' % i)
        cdir = join(folder, 'corpus', 'alphabet_%s' % str(i))
        if not os.path.exists(cdir):
            os.makedirs(cdir)

        print traincsv, cdir
        build_corpus(traincsv, cdir, min_word_size=2, max_word_size=20)
    return


def build_all_corpora(path):
    for folder in get_folders(path):
        build_corpus_for_folder(folder)
    return

if __name__ == '__main__':
    build_all_corpora('NewlyAddedDatasets')

