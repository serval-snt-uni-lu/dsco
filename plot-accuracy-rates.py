import pandas as pd

from pylab import rcParams

import matplotlib.pyplot as plt
from os.path import join
from sklearn.metrics import accuracy_score
from prepare_datasets import get_folders


plt.rc('font', family='sans-serif')


rcParams['figure.figsize'] = 15, 2

all_metrics = []

bdf = pd.DataFrame.from_csv('baseline.csv', index_col=False, header=None)
results_dir = 'final_results_wl_2_to_20'

idx = 0
a = 10
print a
for folder in get_folders('NewlyAddedDatasets/'):
    metrics = {}

    best = 0
    # for a in range(a, a + 1):
    for a in range(3, 21):
        try:
            clsssifiedf = join(folder, results_dir, 'alphabet_%s' % a, 'classified.csv')
            df = pd.DataFrame.from_csv(clsssifiedf, index_col=False)
            y_true, y_pred = df['label'].values.tolist(), df['predicted'].values.tolist()

            accuracy = accuracy_score(y_true, y_pred)
            if accuracy > best:
                best = accuracy
        except Exception as e:
            print e
            pass

    metrics['Dataset'] = bdf.iloc[idx, 0]
    metrics['1NN (Euclidean)'] = 1 - bdf.iloc[idx, 1]
    metrics['1NN (DTW Best Warping)'] = 1 - bdf.iloc[idx, 2]
    metrics['1NN (DTW No Warping)'] = 1 - bdf.iloc[idx, 3]
    metrics['DSCo (Best Alphabet Size)'] = best
    all_metrics.append(metrics)
    idx += 1

df = pd.DataFrame(all_metrics)
df.to_csv('comparison-baseline.csv', index=False)

df = pd.DataFrame.from_csv('comparison-baseline.csv', index_col=False)

print 'DSCo'
print 'Better than DTW'
b = df[df['DSCo (Best Alphabet Size)'] > df['1NN (DTW Best Warping)']].shape[0]
print b, '%.2f' % (100.0 * b / 39)
print 'Better than DTW No Warping'
b = df[df['DSCo (Best Alphabet Size)'] >= df['1NN (DTW No Warping)']].shape[0]
print b, '%.2f' % (100.0 * b / 39)
print 'Better than Euclidean'
b = df[df['DSCo (Best Alphabet Size)'] > df['1NN (Euclidean)']].shape[0]
print b, '%.2f' % (100.0 * b / 39)

print 'DTW'
print 'DTW Better than Euclidean'
b = df[df['1NN (DTW Best Warping)'] > df['1NN (Euclidean)']].shape[0]
print b, '%.2f' % (100.0 * b / 39)

fig = plt.figure()
ndf = df
del ndf['1NN (DTW No Warping)']
del ndf['1NN (Euclidean)']
for i in range(df.shape[0]):
    ndf['Dataset'].iloc[i] = '%s %s' % (str(i+1), ndf['Dataset'].iloc[i])
ndf = ndf.set_index(['Dataset'])

ax = ndf.plot(kind='bar', colormap='Greys')
ax.grid(False)
xticks = ax.get_xticks()
xticks = [x + 0.2 for x in xticks]
ax.set_xticks(xticks)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off
# ax.legend(ncol=3)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=4)

plt.xticks(rotation=35, ha='right')
plt.savefig('/Users/daoyuan.li/Documents/Publications/2015.08.DSCo/AAAI/fig/accuracy-comparison-best.pdf', bbox_inches='tight')
