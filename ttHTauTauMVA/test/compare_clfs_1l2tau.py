import argparse
import numpy as np
import ttHTauTauMVA.ttHTauTauMVA.mva_utils as util
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Compare trained classifiers')
parser.add_argument('-o','--outdir', type=str, default="./", help="output directory")
parser.add_argument('-v','--version', type=str, default="Feb2018")
#parser.add_argument('-r', '--roc', action='store_true',
#                    help="plot ROC curves")
#parser.add_argument('-v', '--validation', action='store_true',
#                    help="plot validation curves")
args = parser.parse_args()

topdir = "/uscms/home/ztao/public_html/BDT"+args.version+"/sklearn/1l2tau/"

clf_dirs = {
    'TT-8Vars':topdir+"matthias/ttbar/",
    'TT-13Vars':topdir+"alexandra/ttbar/",
    'TT-8+2Vars':topdir+"m_extra2vars/ttbar/",
    'TT-8+5Vars':topdir+"m_extra5vars/ttbar/",
    'TT-8Vars+EvisTauAsym':topdir+"m_evisTausAsym/ttbar/",
    'TT-8VarsNew':topdir+"m_repl2vars/ttbar/",
    'TT-13+2Vars':topdir+"a_extra2vars/ttbar/",
    'TT-13+5Vars':topdir+"a_extra5vars/ttbar/",
    'TT-13Vars+EvisTauAsym':topdir+"a_evisTausAsym/ttbar/",
    'TT-13VarsNew':topdir+"a_repl2vars/ttbar/",
    'TTV-13Vars':topdir+"alexandra/ttV/",
    'TTV-13+2Vars':topdir+"a_extra2vars/ttV/",
    'TTV-13+5Vars':topdir+"a_extra5vars/ttV/",
    'TTV-13Vars+EvisTauAsym':topdir+"a_evisTausAsym/ttV/",
    'TTV-13VarsNew':topdir+"a_repl2vars/ttV/",
    'TT-13Vars+EAsym':topdir+"a_extra2vars/ttbar/",
    'TTV-13Vars+EAsym':topdir+"a_extra2vars/ttV/"}


def plot_ROCs(clf_labels, plot_name):

    results=[]
    #clfs=[]
    #labels=[]
    
    for label in clf_labels:
        dir = clf_dirs[label]
        assert(dir.split('/')[-1]=='')
        clf_name = dir+'bdt.pkl'
        dataset_name = dir+'dataset.npz'

        clf = joblib.load(clf_name)
        dataset = np.load(dataset_name)
        
        result = (dataset['y_test'], clf.predict_proba(dataset['x_test'])[:,1],
                  dataset['w_test'], label)
    
        results.append(result)
        #clfs.append(clf)
        #labels.append(label)

    util.plot_rocs(results, args.outdir+'compare_rocs_'+plot_name+'.png', verbose=True, title='')

def plot_ROCs_TestAndTrain(clf_labels, plot_name):

    results=[]

    for label in clf_labels:
        dir = clf_dirs[label]
        assert(dir.split('/')[-1]=='')
        clf_name = dir+'bdt.pkl'
        dataset_name = dir+'dataset.npz'

        clf = joblib.load(clf_name)
        dataset = np.load(dataset_name)

        result_test = (dataset['y_test'], clf.predict_proba(dataset['x_test'])[:,1],
                       dataset['w_test'], label+' test')
        result_train = (dataset['y_train'],
                        clf.predict_proba(dataset['x_train'])[:,1],
                        dataset['w_train'], label+' train')

        results.append(result_train)
        results.append(result_test)
   
    util.plot_rocs(results, args.outdir+'compare_rocs_'+plot_name+'.png', verbose=True, title='')
        

    
plot_ROCs(['TT-8Vars','TT-8+2Vars','TT-8+5Vars','TT-8Vars+EvisTauAsym'],'TT_8Vars')
plot_ROCs(['TT-13Vars','TT-13+2Vars','TT-13+5Vars','TT-13Vars+EvisTauAsym'],'TT_13Vars')

#plot_ROCs(['TTV-8Vars','TTV-8+2Vars','TTV-8+5Vars','TTV-8Vars+EvisTauAsym','TTV-8VarsNew'],'TTV_8Vars')
plot_ROCs(['TTV-13Vars','TTV-13+2Vars','TTV-13+5Vars','TTV-13Vars+EvisTauAsym'],'TTV_13Vars')

plot_ROCs(['TT-8Vars','TT-13Vars','TT-8VarsNew','TT-13VarsNew'],'TT_OldvsNew')
plot_ROCs(['TTV-13Vars','TTV-13VarsNew'],'TTV_OldvsNew')
    
plot_ROCs_TestAndTrain(['TT-13Vars','TT-13Vars+EAsym'],'TT_13Vars_EAsym')
plot_ROCs_TestAndTrain(['TTV-13Vars','TTV-13Vars+EAsym'],'TTV_13Vars_EAsym')

        
#if args.validation:
#    util.plot_validation_curve(clfs,
#                               dataset['x_train'], dataset['x_test'],
#                               dataset['y_train'], dataset['y_test'],
#                               dataset['w_train'], dataset['w_test'],
#                               labels=labels,
#                               figname="compare_validatioin_curves.png")

#if args.roc:
#    util.plot_rocs(results, 'compare_rocs.png')
