import argparse
import numpy as np
import ttHTauTauAnalysis.ttHtautauAnalyzer.mva_utils as util
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Compare trained BDTs')
parser.add_argument('directory', type=str,
                    default='/uscms/home/ztao/nobackup/BDTs/',
                    help="directory where classifiers are stored")
parser.add_argument('-s', '--sklearn', action='store_true',
                    help="include classifiers trained by sklearn")
parser.add_argument('-t', '--tmva', action='store_true',
                    help="include classifiers trained by tmva")
parser.add_argument('-r', '--roc', action='store_true',
                    help="plot ROC curves")
parser.add_argument('-v', '--validation', action='store_true',
                    help="plot validation curves")

args = parser.parse_args()

clfs = []
labels = []
results = []
event_weights = None

if args.sklearn:
    # sklearn weighted
    dir = args.directory+'sklearn/ttV/weighted/'
    name = 'sklearn_weighted'
    #result_sklearn_weighted = util.get_sklearn_test_results(dir, name)
    results.append(util.get_sklearn_test_results(dir, name))

    clf1=joblib.load(dir+'bdt.pkl')
    clfs.append(clf1)
    labels.append('sklearn_weighted')

    dataset = np.load(dir+'dataset.npz')
    x_test = dataset['x_test']
    y_test = dataset['y_test']
    w_test = dataset['w_test']
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    w_train = dataset['w_train']

    event_weights = w_test

    # sklearn unweighted
    dir = args.directory+'sklearn/ttV/unweighted/'
    name = 'sklearn_unweighted'
    #result_sklearn_unweighted = util.get_sklearn_test_results(dir, name)
    results.append(util.get_sklearn_test_results(dir, name))

    clf2=joblib.load(dir+'bdt.pkl')
    clfs.append(clf2)
    labels.append('sklearn_unweighted')

    # sklearn  flip negative weights
    dir = args.directory+'sklearn/ttV/flipnegweight/'
    name = 'sklearn_flip'
    #result_sklearn_flip = util.get_sklearn_test_results(dir, name)
    results.append(util.get_sklearn_test_results(dir, name))

    clf3=joblib.load(dir+'bdt.pkl')
    clfs.append(clf3)
    labels.append('sklearn_flip')

    # sklearn ignore negative weights
    dir = args.directory+'sklearn/ttV/ignorenegweight/'
    name = 'sklearn_ignore'
    #result_sklearn_ignore = util.get_sklearn_test_results(dir, name)
    results.append(util.get_sklearn_test_results(dir, name))

    clf4=joblib.load(dir+'bdt.pkl')
    clfs.append(clf4)
    labels.append('sklearn_ignore')

    # Validation curve
    if args.validation:
        util.plot_validation_curve(clfs, x_train, x_test, y_train, y_test,
                                   w_train, w_test, labels=labels,
                                   figname="compare_validatioin_curves.png")
    
if args.tmva:
    # tmva inverse boost negative weights
    dir = args.directory+'tmva/ttV/weighted/'
    name = 'tmva_weighted'
    #result_tmva_weighted = util.get_tmva_test_results(dir, util.variables_ttV, name)
    results.append(util.get_tmva_test_results(dir, util.variables_ttV, name))

    # tmva unweighted
    dir = args.directory+'tmva/ttV/unweighted/'
    name = 'tmva_unweighted'
    #result_tmva_unweighted = util.get_tmva_test_results(dir, util.variables_ttV, name)
    results.append(util.get_tmva_test_results(dir, util.variables_ttV, name))

    # flip negative weights
    dir = args.directory+'tmva/ttV/flipnegweight/'
    name = 'tmva_flip'
    #result_tmva_flip = util.get_tmva_test_results(dir, util.variables_ttV, name)
    results.append(util.get_tmva_test_results(dir, util.variables_ttV, name))
    
    # ignore negative weights
    dir = args.directory+'tmva/ttV/ignorenegweight/'
    name = 'tmva_ignore'
    #result_tmva_ignore = util.get_tmva_test_results(dir, util.variables_ttV, name)
    results.append(util.get_tmva_test_results(dir, util.variables_ttV, name))

    # pair annihilation
    #dir = args.directory+'tmva/ttV/pairnegweight/'
    #name = 'tmva_pair'
    #result_tmva_pair = util.get_tmva_test_results(dir, util.variables_ttV, name)
    results.append(util.get_tmva_test_results(dir, util.variables_ttV, name))
    
# ROC
if args.roc:
    util.plot_rocs(results, 'compare_rocs.png', weights=event_weights)
    #util.plot_rocs([result_sklearn_weighted, result_sklearn_unweighted, result_sklearn_flip, result_sklearn_ignore, result_tmva_weighted, result_tmva_unweighted, result_tmva_flip, result_tmva_ignore], 'compare_rocs.png', weights=event_weights)
