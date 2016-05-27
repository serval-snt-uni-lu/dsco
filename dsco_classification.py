# Classification
import re, string, random, glob, operator, heapq
from collections import defaultdict
from math import log10

import pandas as pd
from prepare_datasets import get_folders

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from os.path import join
import os


class Segmentor:

    def __init__(self, corpus_1w_file, corpus_2w_file=None, wl=5, N=1024908267229):
        class Pdist(dict):
            "A probability distribution estimated from counts in datafile."
            def __init__(self, data=[], N=N, missingfn=None):
                for key,count in data:
                    self[key] = self.get(key, 0) + int(count)
                ## Number of tokens
                self.N = float(N or sum(self.itervalues()))
                self.missingfn = missingfn or (lambda k, N: 1./N)
            def __call__(self, key):
                if key in self: return self[key]/self.N
                else: return self.missingfn(key, self.N)

        def datafile(name, sep='\t'):
            "Read key,value pairs from file."
            for line in file(name):
                yield line.split(sep)

        def estimate_unknown_word(key, N):
            "Estimate the probability of an unknown word."
            return float(10./(N * 10**len(key)))

        self.wl = wl
        self.Pw = Pdist(datafile(corpus_1w_file), N, estimate_unknown_word)
        if corpus_2w_file:
            self.bigram = True
            self.P2w = Pdist(datafile(corpus_2w_file), N)
        else:
            self.bigram = False


    def product(self, nums):
        "Return the product of a sequence of numbers."
        return sum(log10(num) for num in nums)


    def segment(self, text):
        splits = self.splits(text)
        s = 0
        for i in range(len(splits), 1, -1):
            s += log10(self.cPw(splits[i-1], splits[i-2]))

        return s

    def splits(self, text):
        "Return a list of possible segmentation."
        idx = 0
        text = '^' + text + '$'
        ret = []
        while idx < len(text):
            ret.append(text[idx: idx+self.wl])
            idx += self.wl
        return ret


    def Pwords(self, words):
        "The Naive Bayes probability of a sequence of words."
        return self.product(self.Pw(w) for w in words)

    def cPw(self, word, prev):
        "Conditional probability of word, given previous word."
        try:
            return self.P2w[prev + ' ' + word]/float(self.Pw[prev])
        except KeyError:
            return self.Pw(word)

    def combine(self, Pfirst, first, (Prem, rem)):
        "Combine first and rem results into one (probability, words) pair."
        return Pfirst+Prem, [first]+rem




def calculate_scores(df, label, c1, c2, wl, a, results_dir):
    sg = Segmentor(c1, c2, wl)

    nrows, ncols = df.shape

    scores = []
    for i in range(nrows):
        s = df['sax'].iloc[i]
        score = sg.segment(s)
        scores.append(score)

    df['score_class_%s' % label] = scores
    del df['sax']
    df.to_csv('%s/class_%s.csv' % (results_dir, label), index=False)

    return




def get_dataframe(resultsdir, klasses):
    dfs = []
    for kls in klasses:
        csvf = join(resultsdir, 'class_%s.csv' % kls)
        df = pd.DataFrame.from_csv(csvf, index_col=False)
        if len(dfs):
            del df['label']
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def predict(df):
    df['predicted'] = df.iloc[:, 1:].apply(lambda x: int(x.argmax()[len('score_class_'):]), axis=1)
    return df


def process_results(folder, resultsdir):
    results = []
    for wl in range(2, 21):
        for alphabet in range(3, 21):
            results_dir = join(folder, resultsdir, 'wl_%s_alphabet_%s' % (wl, alphabet))
            traincsv = join(folder, 'train', 'saxified_%s.csv' % alphabet)
            traindf = pd.DataFrame.from_csv(traincsv, index_col=False)
            klasses = sorted(traindf['label'].unique())
            for label in klasses:
                testcsv = join(folder, 'test', 'saxified_%s.csv' % alphabet)
                df = pd.DataFrame.from_csv(testcsv, index_col=False)
                c1 = join(folder, 'corpus', 'alphabet_%s' % alphabet, 'nr_unigram_wl_%s_class_%s.txt' % (wl, label))
                c2 = join(folder, 'corpus', 'alphabet_%s' % alphabet, 'nr_bigram_wl_%s_class_%s.txt' % (wl, label))
                try:
                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)
                except:
                    pass
                calculate_scores(df, label, c1, c2, wl, alphabet, results_dir)
            try:
                clsssifiedf = join(results_dir, 'classified.csv')
                df = get_dataframe(results_dir, klasses)
                df = predict(df)
                df.to_csv(clsssifiedf, index=False)
                accuracy = accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist())
                print 'w: %d\ta: %d\taccuracy: %f' % (wl, alphabet, accuracy)
                results.append((wl, alphabet, accuracy))
            except Exception as e:
                print e

    return results


if __name__ == '__main__':
    for folder in get_folders('NewlyAddedDatasets/'):
        print 'Processing %s' % folder
        print '-' * 80
        results = process_results(folder, resultsdir='results')
        df = pd.DataFrame(results, columns=['WL', 'A', 'Acc'])
        retf = join(folder, 'results.csv')
        df.to_csv(retf, index=False)
        print '-' * 80
        print "DSCo-NG's suggestion:"
        i = df['Acc'].argmax()
        print 'w: %d\ta: %d\taccuracy: %f' % (df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2])
        print 'Results are saved to %s' % retf
        print '-' * 80
