
# coding: utf-8

# In[2]:




# In[1]:

# http://stackoverflow.com/questions/4580877/text-segmentation-dictionary-based-word-splitting
# http://norvig.com/ngrams/ngrams.py

"""
Code to accompany the chapter "Natural Language Corpus Data"
from the book "Beautiful Data" (Segaran and Hammerbacher, 2009)
http://oreilly.com/catalog/9780596157111/

Code copyright (c) 2008-2009 by Peter Norvig

You are free to use this code under the MIT licencse:
http://www.opensource.org/licenses/mit-license.php
"""


import re, string, random, glob, operator, heapq
from collections import defaultdict
from math import log10

import pandas as pd


class Segmentor:
    "A segementation class based on http://norvig.com/ngrams/ngrams.py"

    def memo(f):
        "Memoize function f."
        table = {}
        def fmemo(*args):
            if args not in table:
                table[args] = f(*args)
            return table[args]
        fmemo.memo = table
        return fmemo

    def __init__(self, corpus_1w_file, corpus_2w_file=None, N=1024908267229):
        class Pdist(dict):
            "A probability distribution estimated from counts in datafile."
            def __init__(self, data=[], N=None, missingfn=None):
                for key,count in data:
                    self[key] = self.get(key, 0) + int(count)
                self.N = float(N or sum(self.itervalues()))
                self.missingfn = missingfn or (lambda k, N: 1./N)
            def __call__(self, key):
                if key in self: return self[key]/self.N
                else: return self.missingfn(key, self.N)

        def datafile(name, sep='\t'):
            "Read key,value pairs from file."
            for line in file(name):
                yield line.split(sep)

        def avoid_long_words(key, N):
            "Estimate the probability of an unknown word."
            return 10./(N * 10**len(key))

        self.Pw = Pdist(datafile(corpus_1w_file), N, avoid_long_words)
        if corpus_2w_file:
            self.bigram = True
            self.P2w = Pdist(datafile(corpus_1w_file), N)
        else:
            self.bigram = False


    def product(self, nums):
        "Return the product of a sequence of numbers."
#         return reduce(operator.mul, nums, 1)
        return sum(log10(num) for num in nums)

    @memo
    def segment(self, text, prev='<S>'):
        if self.bigram:
            "Return (log P(words), words), where words is the best segmentation."
            if not text: return 0.0, []
            candidates = [self.combine(log10(self.cPw(first, prev)), first, self.segment(rem, first))
                          for first,rem in self.splits(text)]
            return max(candidates)
        else:
            "Return a list of words that is the best segmentation of text."
            if not text: return []
            candidates = ([first]+self.segment(rem) for first,rem in self.splits(text))
            return max(candidates, key=self.Pwords)

    def splits(self, text, L=20):
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[:i+1], text[i+1:])
                for i in range(min(len(text), L))]

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




def calculate_scores(df, label, c1, c2, results_dir):
    sg = Segmentor(c1, c2)

    nrows, ncols = df.shape

    scores = []
    for i in range(nrows):
        s = df['sax'].iloc[i]
        score, segs = sg.segment(s)
        scores.append(score)

    df['score_class_%s' % label] = scores
    del df['sax']
    df.to_csv('%s/class_%s.csv' % (results_dir, label), index=False)



import sys
from os import listdir
from os.path import isfile, join
import os

if __name__ == '__main__':
#     sys.setrecursionlimit(100000)
    folder = sys.argv[1]
    alphabet = sys.argv[2]
    label = int(sys.argv[3])
    testcsv = join(folder, 'test', 'saxified_%s.csv' % alphabet)
    df = pd.DataFrame.from_csv(testcsv, index_col=False)
    c1 = join(folder, 'corpus', 'alphabet_%s' % alphabet, 'unigram_wl_2_to_20_class_%s.txt' % label)
    c2 = join(folder, 'corpus', 'alphabet_%s' % alphabet, 'bigram_wl_2_to_20_class_%s.txt' % label)
    results_dir = join(folder, 'final_results_wl_2_to_20', 'alphabet_%s' % alphabet)
    try:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    except:
        pass
    calculate_scores(df, label, c1, c2, results_dir)

