
# coding: utf-8

# In[68]:

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from os.path import join
import os
import pandas as pd

from prepare_datasets import get_folders


def get_dataframe(folder, resultsdir, alphabet, klasses):
    resultsdir = join(folder, resultsdir, 'alphabet_%s' % alphabet)
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

def process_results(folder, resultsdir='bgg_results_wl_2_to_20'):
    for a in range(3, 21):
        try:
            clsssifiedf = join(folder, resultsdir, 'alphabet_%s' % a, 'classified.csv')
#                 clsssifiedf = join(folder, 'results_wl_2_to_20', 'alphabet_%s' % a, 'classified.csv')
#             if not os.path.exists(f):
            traincsv = join(folder, 'train', 'saxified_%s.csv' % a)
            traindf = pd.DataFrame.from_csv(traincsv, index_col=False)
            klasses = sorted(traindf['label'].unique())
            df = get_dataframe(folder, resultsdir, a, klasses)
            df = predict(df)
            df.to_csv(clsssifiedf, index=False)
            print folder, a, accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist())
#             print classification_report(df['label'].values.tolist(), df['predicted'].values.tolist())
        except Exception as e:
            print e
            traincsv = join(folder, 'train', 'saxified_%s.csv' % a)
            traindf = pd.DataFrame.from_csv(traincsv, index_col=False)
            if len(traindf.iloc[0, 1]) > 100:
                klasses = sorted(traindf['label'].unique())
                with open('params.txt', 'a') as pf:
                    for kls in klasses:
                        pf.write('%s %s %s\n' % (folder, a, kls))
    print
    return

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from os.path import join
from sklearn.metrics import accuracy_score
from prepare_datasets import get_folders

from pylab import rcParams
import numpy as np


def setAxLinesBW(ax):
    # http://stackoverflow.com/a/7363258
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    COLORMAP = {
        'b': {'marker': None, 'dash': (None,None)},
        'g': {'marker': None, 'dash': [5,5]},
        'r': {'marker': None, 'dash': [5,3,1,3]},
        'c': {'marker': None, 'dash': [1,3]},
        'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
        'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }

    COLORMAP = {
        'b': {'marker': 'o', 'dash': [5,3,1,3]},
        'g': {'marker': 's', 'dash': [5,5]},
        'r': {'marker': '.', 'dash': (None,None)},
        'c': {'marker': 'o', 'dash': [1,3]},
        'm': {'marker': '+', 'dash': [5,2,5,2,5,10]},
        'y': {'marker': '-', 'dash': [5,3,1,2,1,10]},
        'k': {'marker': '.', 'dash': (None,None)} #[1,2,1,10]}
        }

    for line in ax.get_lines():# + ax.get_legend().get_lines():
        origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)
    return


def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)
    return


plt.rc('font', family='sans-serif')


if __name__ == '__main__':
#     process_results('NewlyAddedDatasets/ECG5000/', resultsdir='final_results_wl_2_to_20')
#     process_results('NewlyAddedDatasets/ECG5000/', resultsdir='final_pruned_results_wl_2_to_20')
#     process_results('NewlyAddedDatasets/ECG5000/', resultsdir='final_pruned_mean_results_wl_2_to_20')
    ret = {}
    acc = []
    for a in range(3, 21):
        clsssifiedf = join('NewlyAddedDatasets/ECG5000', 'final_results_wl_2_to_20', 'alphabet_%s' % a, 'classified.csv')
        df = pd.DataFrame.from_csv(clsssifiedf, index_col=False)
        acc.append(accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist()))
        x = {}
        x['BP'] = 0
        x['AP'] = 0
    #         x['APM'] = 0
        for kls in range(1, 6):
            f = 'NewlyAddedDatasets/ECG5000/corpus/alphabet_%s/bigram_wl_2_to_20_class_%s.txt' % (a, kls)
            df = pd.DataFrame.from_csv(f, index_col=False, header=None, sep='\t')
            x['BP'] += df.shape[0]
    #             fp = 'NewlyAddedDatasets/ECG5000/corpus/alphabet_%s/pruned_bigram_wl_2_to_20_class_%s.txt' % (a, kls)
    #             dfp = pd.DataFrame.from_csv(fp, index_col=False, header=None, sep='\t')
    #             x['AP'] += dfp.shape[0]
            fp = 'NewlyAddedDatasets/ECG5000/corpus/alphabet_%s/pruned_mean_bigram_wl_2_to_20_class_%s.txt' % (a, kls)
            dfp = pd.DataFrame.from_csv(fp, index_col=False, header=None, sep='\t')
            x['AP'] += dfp.shape[0]
        ret[a] = x
    df = pd.DataFrame(ret).transpose()
    df.columns = ['# bigrams after pruning', '# bigrams before pruning']


    plt.rc('font', family='sans-serif')
    rcParams['figure.figsize'] = 6, 2

    fig, ax = plt.subplots()


    ax.plot(range(3, 21), acc, c='r', label='Accuracy')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Alphabet size')
    ax.text(.1,.8,'Accuracy',
            horizontalalignment='center',
            fontsize=12,
            transform=ax.transAxes)
    # ax.yaxis.set_ticks(np.arange(0.7, 0.95, 0.5))


    ax1 = ax.twinx()
    # df.plot(ax=ax1, colormap='Greys')
    ax1.plot(range(3, 21), df['# bigrams after pruning'].values.tolist(), label='# bigrams w/o pruning')
    ax1.plot(range(3, 21), df['# bigrams before pruning'].values.tolist(), label='# bigrams w/ pruning')
    ax1.yaxis.set_ticks(np.arange(0, 700000,150000))
    ax1.set_ylabel('# bigrams', fontsize=12)

    ax1.text(.6,.55,'# bigrams without pruning',
            horizontalalignment='center',
            fontsize=12,
            transform=ax1.transAxes)
    ax1.text(.8,.12,'# bigrams with pruning',
            horizontalalignment='center',
            fontsize=12,
            transform=ax1.transAxes)
    setFigLinesBW(fig)
    plt.savefig('/Users/daoyuan.li/Documents/Publications/2015.08.DSCo/AAAI/fig/pruning.pdf', bbox_inches='tight')

