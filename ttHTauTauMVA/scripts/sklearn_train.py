#!/usr/bin/env python

import argparse
from timeit import default_timer as timer

import numpy as np
import ttHTauTauAnalysis.ttHtautauAnalyzer.mva_utils as util
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

parser = argparse.ArgumentParser(description='Train BDT with scikit-learn')
parser.add_argument('background', choices=['ttV','ttbar'],
                    help="background process to train against")
parser.add_argument('ntree', type=int, default=450, nargs='?',
                    help="number of trees")
parser.add_argument('rate', type=float, default=0.02, nargs='?',
                    help="learning rate")
parser.add_argument('depth', type=int, default=3, nargs='?',
                    help="depth of tree")
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
parser.add_argument('-o','--outdir',type=str, default='./',
                    help="output directory")
parser.add_argument('-n','--normalize',action='store_true', #default=False,
                    help="normalize input sample weights")
parser.add_argument('-w','--weights', choices=['u','o','f','z','a'], default='o',
                    help="u: unweighted in training; o: use weights directly from inputs; f: flip all negative weights; z: set all negative weights to zero; a: annihilate pair of negative and positive weighted event (not implemented yet)")

args = parser.parse_args()


# variables used in training
vars = None
if args.background=='ttV':
    vars = util.variables_ttV
elif args.background=='ttbar':
    vars = util.variables_tt

start=timer()
stop=timer()
# Read inputs
if args.timeit:
    start=timer()
if not args.quiet:
    print 'Reading signal samples ...'

xsig, ysig, wsig = util.get_inputs('ttH', vars, dir=args.indir)
if not args.quiet:
    print 'number of signal samples: ', xsig.shape[0]
    print 'Reading background sampels ...'

xbkg, ybkg, wbkg = util.get_inputs(args.background, vars, dir=args.indir)
if not args.quiet:
    print 'number of background samples: ', xbkg.shape[0]

if args.timeit:
    stop=timer()
    print 'Reading inputs takes ', stop-start, 's'

if args.correlation:
    util.plot_correlation(xsig, vars, args.outdir+'correlation_sig.png',
                          verbose=(not args.quiet))
    util.plot_correlation(xsig, vars, args.outdir+'correlation_bkg.png',
                          verbose=(not args.quiet))
    
if args.weights=='u':
    wsig = np.ones(len(wsig))
    wbkg = np.ones(len(wbkg))
elif args.weights=='f':
    wsig = np.array(util.flip_negative_weight(wsig))
    wbkg = np.array(util.flip_negative_weight(wbkg))  
elif args.weights=='z':
    wsig = np.array(util.ignore_negative_weight(wsig))
    wbkg = np.array(util.ignore_negative_weight(wbkg))
#elif args.weights=='o':

if args.normalize:
    wsig *= 1./np.sum(wsig)
    wbkg *= 1./np.sum(wbkg)

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
    print 'Training takes ', stop-start, 's'
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
    util.print_variables_rank(bdt,vars,outname=args.outdir+'ranks.txt',
                             verbose=(not args.quiet))
