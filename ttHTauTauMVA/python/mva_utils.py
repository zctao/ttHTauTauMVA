import numpy as np
import pandas as pd
import ROOT as r
r.gROOT.SetBatch(True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from root_numpy import root2array, rec2array, array2root, fill_hist
from root_numpy.tmva import evaluate_reader
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import learning_curve
#from sklearn.learning_curve import learning_curve
from sklearn.externals import joblib

variables_tt = """
nJet
mindr_lep0_jet
mindr_lep1_jet
avg_dr_jet
max_lep_eta
met
mht
mT_met_lep0
lep0_conept
lep1_conept
dr_leps
dr_lep0_tau
dr_lep1_tau
""".split()

variables_ttV = """
mindr_lep0_jet
mindr_lep1_jet
avg_dr_jet
lep0_conept
lep1_conept
max_lep_eta
mT_met_lep0
dr_leps
mvis_lep0_tau
mvis_lep1_tau
""".split()

# mostly based on:
# https://betatim.github.io/posts/sklearn-for-TMVA-users/
# https://betatim.github.io/posts/advanced-sklearn-for-TMVA/

def read_inputs(file_name, variables, is_signal, tree_name='mva', weight_name='event_weight'):
    # file_name: input ntuple file name; str
    # variables: input variables for training; list of str
    
    x = rec2array(root2array(file_name, tree_name, variables))
    w = root2array(file_name, tree_name, weight_name)
    y = np.ones(x.shape[0]) if is_signal else np.zeros(x.shape[0]) #-1*np.ones(x.shape[0])
    
    return x, y, w

def get_all_variable_names(file_name, tree_name='mva', weight_names=['event_weight']):

    file = r.TFile(file_name)
    tree = file.Get(tree_name)

    var_names = [b.GetName() for b in tree.GetListOfBranches()]

    for wn in weight_names:
        var_names.remove(wn)

    var_names.remove("isGenMatchedTau")
    var_names.remove("HiggsDecayType")
    var_names.remove("run")
    var_names.remove("lumi")
    var_names.remove("nEvent")
    
    return var_names
    

def combine_inputs(datatuples, xsections=None, lumi=1.):
    # datatuples: list of data (x, y, w)
    # xsections: list of cross sections for the corresponding samples

    x = None
    y = None
    w = None

    for i, data in enumerate(datatuples):
        xi, yi, wi = data

        if xsections is not None:
            assert(len(xsections)==len(datatuples))
            # scale sample weights based on integrated luminosity and cross section
            wi *= lumi * xsections[i] / np.sum(wi)  #?
        
        if x is None:
            x = xi
            y = yi
            w = wi
        else:
            x = np.concatenate((x, xi))
            y = np.concatenate((y, yi))
            w = np.concatenate((w, wi))

    return x, y, w
    

def get_inputs(sample_name,variables,filename=None,tree_name='mva',dir='',
               weight_name='event_weight', lumi=1.):
    x = None
    y = None
    w = None

    infiles = []
    xsections = []
    if filename!=None:
        infiles = [dir+filename]
    else:
        if ('ttH' in sample_name):
            infiles = [dir+"mvaVars_ttH_loose.root"]
            xsections = [0.215]
        elif ('ttV' in sample_name):
            infiles = [dir+"mvaVars_TTZ_loose.root", dir+"mvaVars_TTW_loose.root"]
            xsections = [0.253, 0.204]  # [TTZ, TTW]
        elif ('ttbar' in sample_name):
            infiles = [dir+"mvaVars_TTSemilep_loose.root",
                       dir+"mvaVars_TTDilep_loose.root"]
            xsections = [182, 87.3]  # [semilep, dilep]
        else:
            print "Pick one sample name from 'ttH', 'ttV' or 'ttbar'"
            return x, y, w

    for fn, xs in zip(infiles, xsections):
        xi = rec2array(root2array(fn, tree_name, variables))
        wi = root2array(fn, tree_name, weight_name)

        # scale weight and renormalize total weights to one
        #wi *= (xs / sum(xsections)) /np.sum(wi)
        
        # scale samples based on lumi and cross section
        wi *= lumi * xs / np.sum(wi)

        if x is not None:
            x = np.concatenate((x,xi))
            w = np.concatenate((w,wi))
        else:
            x = xi
            w = wi

    if ('ttH' in sample_name):
        y = np.ones(x.shape[0])
    else:
        y = np.zeros(x.shape[0])
        #y = -1*np.ones(x.shape[0])
    
    return x, y, w


#def dump_dataset(data,split,filename='dataset.root',dir='',mode='recreate'):
#    x, y, w = data
#    array2root(x[y>0.5],filename,treename=split+'/signal', mode=mode)
#    array2root(w[y>0.5],filename,treename=split+'/signal', mode='update')
#    array2root(x[y<0.5],filename,treename=split+'/background', mode='update')
#    array2root(w[y<0.5],filename,treename=split+'/background', mode='update')
    
def update_weights(weights, wtype):
    if wtype=='u':
        return np.ones(len(weights))
    elif wtype=='f':
        return np.array(util.flip_negative_weight(weights))
    elif wtype=='z':
        return np.array(util.ignore_negative_weight(weights))
    else:
        return weights

#def correct_negative_weight(weights):
#    # based on the suggestion from @glouppe
#    # https://github.com/scikit-learn/scikit-learn/issues/3774
#    weights_corrected = []
#    for w in weights:
#        if w < 0:
#
#        else:
#            weights_corrected.append(w)
def flip_negative_weight(weights):
    w_flip = []
    for w in weights:
        w_flip.append(abs(w))
    return w_flip
            
def ignore_negative_weight(weights):
    w_new = []
    for w in weights:
        w_new.append(w if w>0. else 0.)
    return w_new


def plot_correlation(x, variables, figname, verbose=False, **kwds):
    df = pd.DataFrame(x, columns=variables)
    
    """Calculate pairwise correlation between variables"""
    corrmat = df.corr(**kwds)

    fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))

    opts = {'cmap': plt.get_cmap("RdBu"),'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    ax1.set_title("Correlations")

    labels = corrmat.columns.values
    for ax in (ax1,) :
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

    if verbose:
        print 'Generate plot : ',figname
    
    
def plot_roc(data, figname, verbose=False):
    # data: tuple (test, pred, weights)
    y_test, y_pred, w_test = data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, sample_weight=w_test)
    #print max(thresholds) #print min(thresholds)
    roc_auc = roc_auc_score(y_test, y_pred, sample_weight=w_test)
    #roc_auc = auc(fpr, tpr,reorder=True)
    
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %.03f)'%(roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Background efficiency')
    plt.ylabel('Signal efficiency')
    plt.legend(loc="lower right")
    plt.grid()
    #plt.show()
    plt.savefig(figname)
    plt.close()

    if verbose:
        print 'Generate plot : ', figname


def plot_rocs(data_list, figname, verbose=False, weights=None,
              title="Receiver Operating Characteristic Curve"):
    # data_list is expected to be a list of tuple (y_test, y_pred, w_test, label)
    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Background efficiency')
    plt.ylabel('Signal efficiency')
    for data in data_list:
        y_test, y_pred, w_test, label = data

        # use specified weights to plot roc curve if weights is provided
        if not weights is None:
            w_test = weights
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, sample_weight=w_test)
        roc_auc = roc_auc_score(y_test, y_pred, sample_weight=w_test)
        plt.plot(fpr, tpr, lw=1, label=label+' (area = %.03f)'%(roc_auc))

    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(figname)
    plt.close()

    if verbose:
        print 'Generate plot : ', figname
    
def plot_clf_results(decisions, weights, nbins=30, figname="BDTOutput.png",
                     verbose="False"):

    # unpack np array tuple
    decision_sig_train, decision_bkg_train, decision_sig_test, decision_bkg_test = decisions
    for w in weights:
        w *= 1./np.sum(w)
    weight_sig_train, weight_bkg_train, weight_sig_test, weight_bkg_test = weights
    
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)

    h_train_sig = r.TH1D('train_sig','',nbins,low,high)
    h_train_sig.SetStats(0)
    fill_hist(h_train_sig, decision_sig_train, weights=weight_sig_train)

    h_train_bkg = r.TH1D('train_bkg','',nbins,low,high)
    h_train_bkg.SetStats(0)
    fill_hist(h_train_bkg, decision_bkg_train, weights=weight_bkg_train)

    h_test_sig = r.TH1D('test_sig','',nbins,low,high)
    h_test_sig.SetStats(0)
    fill_hist(h_test_sig, decision_sig_test, weights=weight_sig_test)

    h_test_bkg = r.TH1D('test_bkg','',nbins,low,high)
    h_test_bkg.SetStats(0)
    fill_hist(h_test_bkg, decision_bkg_test, weights=weight_bkg_test)

    # legend
    l = r.TLegend(0.75,0.75,0.9,0.9)
    l.AddEntry(h_train_sig,'S (train)','f')
    l.AddEntry(h_train_bkg,'B (train)','f')
    l.AddEntry(h_test_sig,'S (test)','p')
    l.AddEntry(h_test_bkg,'B (test)','p')
    
    # draw histograms
    ymax = max(h_train_sig.GetMaximum(), h_train_bkg.GetMaximum(),
               h_test_sig.GetMaximum(), h_test_bkg.GetMaximum())
    
    tcanvas = r.TCanvas()
    h_train_sig.SetLineColor(2)
    h_train_sig.SetFillColorAlpha(2,0.5)
    h_train_sig.GetXaxis().SetTitle('BDT Output')
    h_train_sig.GetYaxis().SetTitle('A.U.')
    h_train_sig.SetMaximum(ymax*1.2)
    h_train_sig.Draw("HIST")
    h_train_bkg.SetLineColor(4)
    h_train_bkg.SetFillColorAlpha(4,0.5)
    h_train_bkg.Draw('SAME HIST')
    h_test_sig.SetMarkerStyle(20)
    h_test_sig.SetMarkerColor(2)
    h_test_sig.SetMarkerSize(0.8)
    h_test_sig.SetLineColor(2)
    h_test_sig.Draw('SAME')
    h_test_bkg.SetMarkerStyle(20)
    h_test_bkg.SetMarkerColor(4)
    h_test_bkg.SetMarkerSize(0.8)
    h_test_bkg.SetLineColor(4)
    h_test_bkg.Draw('SAME')
    l.Draw('SAME')

    tcanvas.SaveAs(figname)

    if verbose:
        print 'Generate plot : ', figname


def plot_clf_results_tmva(reader, x_train, y_train, w_train, x_test, y_test, w_test, nbins=30, figname="BDTOutput_tmva.png", verbose="False"):

    decisions = []
    weights = []
    for x,y,w in ((x_train, y_train, w_train), (x_test, y_test, w_test)):
        w *= 1./np.sum(w)
        dsig = evaluate_reader(reader, "BDT", x[y>0.5])
        wsig = w[y>0.5]
        dbkg = evaluate_reader(reader, "BDT", x[y<0.5])
        wbkg = w[y<0.5]
        decisions += [dsig, dbkg]
        weights += [wsig, wbkg]

    plot_clf_results(tuple(decisions), tuple(weights), nbins, figname, verbose)
        
def plot_clf_results_sklearn(clf, x_train, y_train, w_train, x_test, y_test, w_test, nbins=30, figname="BDTOutput_sklearn.png", verbose="False"):
    
    decisions = []
    weights = []
    for x,y,w in ((x_train, y_train, w_train), (x_test, y_test, w_test)):
        w *= 1./np.sum(w)
        #dsig = clf.decision_function(x[y>0.5])
        dsig = clf.predict_proba(x[y>0.5])[:,1]
        wsig = w[y>0.5]
        #dbkg = clf.decision_function(x[y<0.5])
        dbkg = clf.predict_proba(x[y<0.5])[:,1]
        wbkg = w[y<0.5]
        decisions += [dsig,dbkg]
        weights += [wsig, wbkg]

    plot_clf_results(tuple(decisions), tuple(weights), nbins, figname, verbose)
        
    
def print_variables_rank(clf, variables, outname=None, verbose=False):

    out = None
    if not outname==None:
        out = open(outname, 'w')
    elif verbose:
        print 'classifier variable ranking : '
        
    for var, score in sorted(zip(variables, clf.feature_importances_),key=lambda x: x[1], reverse=True):
        if out==None:
            print var,'\t',score
        else:
            out.write(str(var)+'\t'+str(score)+'\n')

    if not out==None:
        out.close()
        if verbose:
            print 'Output : ', outname

        
def run_grid_search(clf, x_dev, y_dev, w_dev,
                    param_grid = {"n_estimators": [50,200,400],
                                  "max_depth": [1, 3, 5],
                                  'learning_rate': [0.1, 0.2, 1.]},
                    verbose=0):

    clfGS = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', n_jobs=8,
                         verbose=verbose-1)
    clfGS.fit(x_dev,y_dev, w_dev)

    print 'Best parameters set found on development set: '
    print clfGS.best_estimator_

    if verbose:
        print
        print 'Grid scores on a subset of the development set:'
        means = clfGS.cv_results_['mean_test_score']
        stds = clfGS.cv_results_['std_test_score']
        params = clfGS.cv_results_['params']
        for mean, std, params in zip(means, stds, params):
            print "%0.4f (+/-%0.04f) for %r"%(mean, std * 2, params)
        
        #y_true, y_pred = y_dev, clf.decision_function(x_dev)
        #print "  It scores %0.4f on the full development set"%roc_auc_score(y_true, y_pred)
        #y_true, y_pred = y_eval, clf.decision_function(x_eval)
        #print "  It scores %0.4f on the full evaluation set"%roc_auc_score(y_true, y_pred)

        
def plot_validation_curve(clfs, x_train, x_test, y_train, y_test, w_train, w_test,
                          labels=None, figname="validation_curve.png"):
    for n,clf in enumerate(clfs):
        test_score = np.empty(len(clf.estimators_))
        train_score = np.empty(len(clf.estimators_))
        
        #for i, pred in enumerate(clf.staged_decision_function(x_test)):
        #    test_score[i] = roc_auc_score(y_test, pred, sample_weight=w_test)
        for i, pred in enumerate(clf.staged_predict_proba(x_test)):
            test_score[i] = roc_auc_score(y_test, pred[:,1], sample_weight=w_test)

        #for i, pred in enumerate(clf.staged_decision_function(x_train)):
        #    train_score[i] = roc_auc_score(y_train, pred, sample_weight=w_train)
        for i, pred in enumerate(clf.staged_predict_proba(x_train)):
            train_score[i] = roc_auc_score(y_train, pred[:,1], sample_weight=w_train)
        
        best_iter = np.argmax(test_score)
        rate = clf.get_params()['learning_rate']
        depth = clf.get_params()['max_depth']
        
        label='rate=%.2f depth=%i (%.2f)'%(rate,depth,test_score[best_iter])

        if not labels is None:
            label = labels[n]
            
        test_line = plt.plot(test_score,label=label)
        colour = test_line[-1].get_color()
        plt.plot(train_score, '--', color=colour)
        plt.xlabel("Number of boosting iterations")
        plt.ylabel("Area under ROC")
        plt.axvline(x=best_iter, color=colour)

    plt.legend(loc='best')
    #return plt
    plt.savefig(figname)
    plt.close()

    
def plot_learning_curve(estimator, title, X, y, ylim=None,
                        cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None, xlabel=True):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    figname = title.replace(' ','_')+'.png'
    plt.savefig(figname)
    plt.close()
    
#def get_sklearn_test_results(directory, name=''):
#    # load dataset and classifier
#    clf = joblib.load(directory+'bdt.pkl')
#    dataset = np.load(directory+'dataset.npz')
#
#    result = (dataset['y_test'], clf.predict_proba(dataset['x_test'])[:,1],
#              dataset['w_test'], name)
#    return result

def get_sklearn_test_results(clf_name, dataset_name, label=''):
    # load dataset and classifier
    clf = joblib.load(clf_name)
    dataset = np.load(dataset_name)

    result = (dataset['y_test'], clf.predict_proba(dataset['x_test'])[:,1],
              dataset['w_test'], label)
    return result

def get_tmva_test_results(directory, variables, name=''):
    # TMVA reader 
    reader = r.TMVA.Reader()
    for var in variables:
        #vtype = 'i' if var in ['nJet','tau0_decaymode','tau1_decaymode','ntags','ntags_loose'] else 'f'
        reader.AddVariable(var, array('f', [0]))

    reader.BookMVA('BDT',directory+'weights/TMVA_BDT.weights.xml')

    # Get testing dataset
    filename = directory+'tmva_output.root'
    x_test = rec2array(root2array(filename, 'TestTree', variables))
    y_test = 1-root2array(filename, 'TestTree', 'classID')
    w_test = root2array(filename, 'TestTree', 'weight')
    
    y_decision = evaluate_reader(reader, "BDT", x_test)
    return (y_test, y_decision, w_test, name)
