#!/usr/bin/env python

import argparse
from timeit import default_timer as timer

import numpy as np
import ttHTauTauMVA.ttHTauTauMVA.mva_utils as util
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Train BDT with scikit-learn')
parser.add_argument('background', choices=['ttV','ttbar'],
                    help="background process to train against")
parser.add_argument('anatype', choices=['1l2tau','2lss1tau','3l1tau'])
parser.add_argument('ntree', type=int, default=450, nargs='?',
                    help="number of trees")
parser.add_argument('rate', type=float, default=0.02, nargs='?',
                    help="learning rate")
parser.add_argument('depth', type=int, default=3, nargs='?',
                    help="depth of tree")
parser.add_argument('-vars', '--variables', nargs='+', type=str,
                    help="List of input variables for training. If none provided, take all available branches from input root tree")
parser.add_argument('-e', '--evaluate', action='store_true',
                    help="evaluate training results")
parser.add_argument('-c', '--correlation', action='store_true',
                    help='plot correlation of input variables')
parser.add_argument('-t','--timeit', action='store_true',
                    help="use timer")
#parser.add_argument('-v','--verbose',action='store_true',
#                    help="verbosity")
parser.add_argument('-q','--quiet',action='store_true',
                    help="quiet mode")
parser.add_argument('-i','--indir',type=str, default='./',
                    help="input directory")
parser.add_argument('--ntuple_prefix', type=str, default='mvaVars_',
                    help="Prefix of mva ntuple files")
parser.add_argument('-o','--outdir',type=str, default='./',
                    help="output directory")
#parser.add_argument('-n','--normalize',action='store_true', #default=False,
#                    help="normalize input sample weights")
parser.add_argument('-w','--weights', choices=['u','o','f','z','a'], default='o',
                    help="u: unweighted in training; o: use weights directly from inputs; f: flip all negative weights; z: set all negative weights to zero; a: annihilate pair of negative and positive weighted event (not implemented yet)")

args = parser.parse_args()

###################
# Input files

# signal input file
# ttH non bb
infile_sig = args.indir+args.ntuple_prefix+"ttH_"+args.anatype+".root"
xs_sig = 0.215 # ttHnonbb
# background input file
infiles_bkg = []
xs_bkg = []
if args.background == "ttV":
    # TTW
    infiles_bkg.append(args.indir+args.ntuple_prefix+"TTW_"+args.anatype+".root")
    xs_bkg.append(0.204)
    # TTZ
    infiles_bkg.append(args.indir+args.ntuple_prefix+"TTZ_"+args.anatype+".root")
    xs_bkg.append(0.253)
elif args.background == "ttbar":
    infiles_bkg.append(args.indir+args.ntuple_prefix+"TT_SemiLep_"+args.anatype+".root")
    xs_bkg.append(182.)
    infiles_bkg.append(args.indir+args.ntuple_prefix+"TT_DiLep_"+args.anatype+".root")
    xs_bkg.append(87.3)
else:
    print "Invalid background. Choose between 'ttV' and 'ttbar' for background type."
    exit()

# variables used in training
var = util.get_all_variable_names(infile_sig) if args.variables is None else args.variables

print "Input variables for training:",var

start=timer()
stop=timer()
#################
# Read inputs

def update_weights(weights, wtype):
    if wtype=='u':
        return np.ones(len(weights))
    elif wtype=='f':
        return np.array(util.flip_negative_weight(weights))
    elif wtype=='z':
        return np.array(util.ignore_negative_weight(weights))
    else:
        return weights
    

if args.timeit:
    start=timer()
if not args.quiet:
    print 'Reading signal samples ...'
    
xsig, ysig, wsig = util.read_inputs(infile_sig, var, True)
wsig = update_weights(wsig, args.weights)
# scale
wsig *= 1. * xs_sig / np.sum(wsig)

if not args.quiet:
    print 'number of signal samples: ', xsig.shape[0]
    print 'Reading background sampels ...'

dataset_bkg = []
for in_bkg in infiles_bkg:
    xbi, ybi, wbi = util.read_inputs(in_bkg, var, False)
    wbi = update_weights(wbi, args.weights)
    dataset_bkg.append((xbi, ybi, wbi))

# combine background samples
xbkg, ybkg, wbkg = util.combine_inputs(dataset_bkg, xs_bkg, 1.)

if not args.quiet:
    print 'number of background samples: ', xbkg.shape[0]

if args.timeit:
    stop=timer()
    print 'Reading inputs took ', stop-start, 's'

# finish reading inputs
    
if args.correlation:
    util.plot_correlation(xsig, var, args.outdir+'correlation_sig.png',
                          verbose=(not args.quiet))
    util.plot_correlation(xsig, var, args.outdir+'correlation_bkg.png',
                          verbose=(not args.quiet))

x = np.concatenate((xsig, xbkg))
y = np.concatenate((ysig, ybkg))
w = np.concatenate((wsig, wbkg))

x_train,x_test,y_train,y_test,w_train,w_test = train_test_split(x, y, w,
                                                                test_size=0.2,
                                                                #train_size=10000,
                                                                #test_size=5000,
                                                                random_state=0)

# Training
if not args.quiet:
    print 'start training ...'
if args.timeit:
    start = timer()

bdt = GradientBoostingClassifier(n_estimators=args.ntree, learning_rate=args.rate,
                                 max_depth=args.depth, random_state=0)
bdt.fit(x_train, y_train, w_train)

# save output
outfile=args.outdir+'bdt.pkl'
joblib.dump(bdt, outfile, compress=True)
if not args.quiet:
    print 'save classifier to ', outfile

# save dataset
outdata=args.outdir+'dataset'
np.savez(outdata, x_train=x_train, y_train=y_train, w_train=w_train,
         x_test=x_test, y_test=y_test, w_test=w_test)
if not args.quiet:
    print 'save datasets to ', outdata+'.npz'


#util.dump_dataset((x_train, y_train, w_train), 'train', filename=outdata,
#                  mode='recreate')
#util.dump_dataset((x_test, y_test, w_test), 'test', filename=outdata,
#                  mode='update')

if args.timeit:
    stop = timer()
    print 'Training took ', stop-start, 's'
if not args.quiet:
    print 'Training done'
    print 'Output : ', outfile

# evaluate training results
if args.evaluate:
    #y_pred = bdt.decision_function(x_test)#.ravel()
    y_pred = bdt.predict_proba(x_test)[:,1]
    util.plot_clf_results(bdt, x_train, y_train, w_train, x_test, y_test, w_test,
                          figname=args.outdir+"bdtoutput.png",
                          verbose=(not args.quiet))
    util.plot_roc((y_test, y_pred, w_test), figname=args.outdir+'roc.png',
                  verbose=(not args.quiet))
    util.print_variables_rank(bdt,var,outname=args.outdir+'ranks.txt',
                             verbose=(not args.quiet))
