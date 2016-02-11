# coding: utf-8

get_ipython().magic(u'matplotlib inline')
from prepare_datasets import get_folders
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from matplotlib import pyplot as plt


from pylab import rcParams
rcParams['figure.figsize'] = 9, 6

rcParams['legend.loc'] = 'best'
rcParams['image.cmap'] = 'Greys'


import numpy as np

from matplotlib import gridspec

def setAxLinesBW(ax):
    # http://stackoverflow.com/a/7363258
    """
    Take each Line2D in the axes, ax, and convert the line style to be
    suitable for black and white viewing.
    """
    MARKERSIZE = 2

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
        'b': {'marker': None, 'dash': (None,None)},
        'g': {'marker': 's', 'dash': [5,5]},
        'r': {'marker': '*', 'dash': [5,3,1,3]},
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



fig, axs = plt.subplots(8, 5, sharex='col', sharey='row')
fig.subplots_adjust(hspace=0.15, wspace=.1)
axs = axs.ravel()

# print dir(axs), type(axs)
i = 0
j = 0
good = 0
for d in get_folders('NewlyAddedDatasets/'):
    txt = d[len('NewlyAddedDatasets/'):]
    r = []
    for a in range(3, 21):
        f = os.path.join(d, 'results_wl_2_to_20', 'alphabet_%s' % a, 'classified.csv')
        df = pd.DataFrame.from_csv(f, index_col=False)
        r.append(accuracy_score(df['label'].values.tolist(), df['predicted'].values.tolist()))
    ax = axs[j]
    j += 1
    df = pd.DataFrame.from_csv(os.path.join(d, 'sax_1nn.csv'), index_col=False)
    df = df.set_index('alphabet')
    ax.plot(range(3, 21), df['accuracy'].values.tolist(), label='%s' % j)
    ax.plot(range(3, 21), r, 'r-')
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.trapz.html
    # http://stackoverflow.com/a/13323861
    a1 = np.trapz(df['accuracy'].values.tolist())
    a2 = np.trapz(r)
    if a2 > a1:
        good += 1
    ax.set_xlim([3, 20])
    ax.xaxis.set_ticks(np.arange(3, 21, 5))

    ax.set_ylim([0, 1])
    ax.yaxis.set_ticks(np.arange(0, 1.2, 0.5))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.text(3.5, 0.2, txt, fontsize=6)
    ax.text(3.5, 0.5, '%.2f %.2f' % (a1, a2), fontsize=6)


ax = axs[-1]
ax.set_xlim([3, 20])
ax.xaxis.set_ticks(np.arange(3, 21, 5))

ax.set_ylim([0, 1])
ax.yaxis.set_ticks(np.arange(0, 1.2, 0.5))
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=7)
plt.savefig('/Users/daoyuan.li/Documents/Papers/2016.01.MLDM/fig/sax-dsco.pdf', bbox_inches='tight')
print good
