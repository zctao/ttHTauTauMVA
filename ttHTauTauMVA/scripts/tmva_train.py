#!/usr/bin/env python

import argparse
from timeit import default_timer as timer
import numpy as np
from array import array
import ttHTauTauAnalysis.ttHtautauAnalyzer.mva_utils as util
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from ROOT import TMVA, TFile, TCut
from root_numpy.tmva import add_classification_events, evaluate_reader

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
parser.add_argument('-w','--weights', choices=['u','o','f','z'], default='o',
                    help="u: unweighted in training; o: use weights directly from inputs; f: flip all negative weights; z: set all negative weights to zero")
parser.add_argument('--tmva_negweight', choices=['InverseBoostNegWeights','IgnoreNegWeightsInTraining','PairNegWeightsGlobal'], default='IgnoreNegWeightsInTraining',help="TMVA negative weight treatment")

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
    
TMVA.gConfig().GetIONames().fWeightFileDir = args.outdir+'weights/'
output = TFile(args.outdir+'tmva_output.root', 'recreate')
factory = TMVA.Factory('TMVA', output, 'AnalysisType=Classification:'
                       '!V:Silent:!DrawProgressBar')
for var in vars:
    vtype = 'I' if var=='nJet' else 'F'
    factory.AddVariable(var, vtype)

add_classification_events(factory, x_train, y_train, weights=w_train)
add_classification_events(factory, x_test, y_test, weights=w_test, test=True)

norm = 'None'
#if args.normalize:
#    norm = 'NumEvents'
factory.PrepareTrainingAndTestTree(TCut('1'), 'NormMode={}'.format(norm))

config="NTrees={}:MaxDepth={}:BoostType=Grad:SeparationType=GiniIndex:Shrinkage={}:NegWeightTreatment={}".format(args.ntree,args.depth,args.rate,args.tmva_negweight)

if not args.quiet:
    print config

factory.BookMethod(TMVA.Types.kBDT, "BDT", config)
factory.TrainAllMethods()

if args.timeit:
    stop = timer()
    print 'Training takes ', stop-start, 's'
if not args.quiet:
    print 'Training done'
    print 'Outputs in directory : ', args.outdir

# evaluate training results
if args.evaluate:
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    reader = TMVA.Reader()
    for var in vars:
        vtype = 'i' if var=='nJet' else 'f'
        reader.AddVariable(var, array(vtype, [0.]))

    reader.BookMVA('BDT',args.outdir+'weights/TMVA_BDT.weights.xml')
    y_decision = evaluate_reader(reader, "BDT", x_test)

    util.plot_roc((y_test, y_decision, w_test),figname=args.outdir+'roc.png')
    # TODO
    # variable rank
    # plot_clf_results
