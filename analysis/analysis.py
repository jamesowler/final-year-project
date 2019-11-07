import glob
import numpy as np
from scipy.io import loadmat

import Python.bayesiantests as bst

def correlated_ttest_(n_folds, classifier1, classifier2):


    results_dir = r'C:\Users\James\Projects\final-year-project\Wei-Sam\results'

    class1_acc = []
    class2_acc = []

    class1_roc = []
    class2_roc = []

    # load data for classifier 1
    for i in range(n_folds):
        results = results_dir + f'\{classifier1}_{str(i)}'
        files = glob.glob(f'{results}\*')
        test_files = []

        [test_files.append(i) for i in files if 'test' in i]
        
        for j in test_files:
            
            stats_file = j + '\stats.mat'
            stats = loadmat(stats_file)

            accuray = stats['stats'][0][2]
            roc = stats['stats'][0][3]
            class1_acc.append(accuray)
            class1_roc.append(roc)
    
    # load data for classifier 2
    for i in range(n_folds):
        results = results_dir + f'\{classifier2}_{str(i)}'
        files = glob.glob(f'{results}\*')
        test_files = []

        [test_files.append(i) for i in files if 'test' in i]
        
        for j in test_files:
            stats_file = j + '\stats.mat'
            stats = loadmat(stats_file)

            accuray = stats['stats'][0][2]
            roc = stats['stats'][0][3]
            class2_acc.append(accuray)
            class2_roc.append(roc)

    accuray_diff = np.array(class1_acc) - np.array(class2_acc)
    roc_diff = np.array(class1_roc) - np.array(class2_roc)

    # print(accuray_diff, roc_diff)

    p_acc = bst.correlated_ttest(accuray_diff, 0.003,  runs=n_folds)
    p_roc = bst.correlated_ttest(roc_diff, 0.003, runs=n_folds)

    print(np.sum(accuray_diff))
    print(p_acc, p_roc)

if __name__ == '__main__':
    correlated_ttest_(5, 'mixed_drive_n4_gmm', 'mixed_drive_gmm')
