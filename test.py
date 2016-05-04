from prepare_datasets import get_folders
from prepare_datasets import saxify_all
from build_corpus import build_all_corpora
from dsco_classification import process_results
from os.path import join
import pandas as pd


if __name__ == '__main__':

    print 'Converting .mat to .csv and Symbolizing real-valued data to strings'
    saxify_all('NewlyAddedDatasets')
    print 'Done converting datasets'
    print '-' * 80

    print 'Extracting unigrams and bigrams'
    build_all_corpora('NewlyAddedDatasets')
    print 'Done extracting unigrams and bigrams'
    print '-' * 80

    for folder in get_folders('NewlyAddedDatasets/'):
        print 'Processing %s' % folder
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
