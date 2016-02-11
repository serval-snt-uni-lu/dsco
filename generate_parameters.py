
# coding: utf-8

# In[1]:

import pandas as pd
from os.path import join

def generate_parameters(folder, alphabet_range, reset=False, pfile='params.txt'):
    df = pd.DataFrame.from_csv(join(folder, 'train', 'saxified_10.csv'), index_col=False)
    klasses = sorted(df['label'].unique())
    mod = 'a'
    if reset:
        mod = 'w'
    with open(pfile, mod) as f:
        for k in klasses:
            for a in range(alphabet_range[0], alphabet_range[1] + 1):
                f.write('%s %s %s\n' % (folder, a, k))
    return


# generate_parameters('NewlyAddedDatasets/ElectricDevices/', (5, 10), reset=True, pfile='params.txt')
# generate_parameters('NewlyAddedDatasets/LargeKitchenAppliances', (3, 20), reset=True, pfile='params1.txt')
# generate_parameters('NewlyAddedDatasets/SmallKitchenAppliances', (3, 20), reset=False, pfile='params1.txt')
# generate_parameters('NewlyAddedDatasets/RefrigerationDevices', (3, 20), reset=False, pfile='params1.txt')

# generate_parameters('NewlyAddedDatasets/ECG5000/', (5, 10), reset=True, pfile='params.txt')
# generate_parameters('Pre_Summer_2015_Datasets/OSULeaf/', (5, 10), reset=True, pfile='params_leaf')
# generate_parameters('Pre_Summer_2015_Datasets/SwedishLeaf/', (5, 10), reset=True, pfile='params_leaf')

from prepare_datasets import get_folders

for folder in get_folders('NewlyAddedDatasets/'):
#     generate_parameters(folder, (3, 20))
#     generate_parameters(folder, (8, 10))
    generate_parameters(folder, (3, 20))


# In[5]:

s = '''NewlyAddedDatasets/Wine
NewlyAddedDatasets/Strawberry
NewlyAddedDatasets/ArrowHead
NewlyAddedDatasets/InsectWingbeatSound
NewlyAddedDatasets/WordSynonyms
NewlyAddedDatasets/ToeSegmentation1
NewlyAddedDatasets/ToeSegmentation2
NewlyAddedDatasets/Ham
NewlyAddedDatasets/Meat
NewlyAddedDatasets/FordA
NewlyAddedDatasets/FordB
NewlyAddedDatasets/ShapeletSim
NewlyAddedDatasets/BeetleFly
NewlyAddedDatasets/BirdChicken
NewlyAddedDatasets/Earthquakes
NewlyAddedDatasets/Herring
NewlyAddedDatasets/ShapesAll
NewlyAddedDatasets/Computers
NewlyAddedDatasets/LargeKitchenAppliances
NewlyAddedDatasets/RefrigerationDevices
NewlyAddedDatasets/ScreenType
NewlyAddedDatasets/SmallKitchenAppliances
NewlyAddedDatasets/Worms
NewlyAddedDatasets/WormsTwoClass
NewlyAddedDatasets/UWaveGestureLibraryAll
NewlyAddedDatasets/Phoneme
NewlyAddedDatasets/HandOutlines'''

i = 1
for line in s.split('\n'):
    generate_parameters(line.strip(), (10, 20))
    i += 1
print i

