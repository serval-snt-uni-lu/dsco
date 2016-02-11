
# coding: utf-8

# In[16]:

import pandas as pd

for a in range(3, 21):
    for kls in range(1, 6):
        f = 'NewlyAddedDatasets/ECG5000/corpus/alphabet_%s/bigram_wl_2_to_20_class_%s.txt' % (a, kls)

        df = pd.DataFrame.from_csv(f, index_col=False, header=None, sep='\t')
        df.columns = ['k', 'f']
    #     print i, df['f'].describe()
        m = df['f'].min()
        print a, kls, df.shape[0], df[df['f'] > m].shape[0]
        df[df['f'] > mean].to_csv('NewlyAddedDatasets/ECG5000/corpus/alphabet_%s/pruned_bigram_wl_2_to_20_class_%s.txt' % (a, kls), index=False, header=None, sep='\t')
    print

