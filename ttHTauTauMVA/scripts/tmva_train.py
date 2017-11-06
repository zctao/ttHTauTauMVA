#!/usr/bin/env python

import argparse
from timeit import default_timer as timer
import numpy as np
from array import array
import ttHTauTauMVA.ttHTauTauMVA.mva_utils as util
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from ROOT import TMVA, TFile, TCut
from root_numpy.tmva import add_classification_events, evaluate_reader

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
parser.add_argument('-n','--normalize',action='store_true', #default=False,
                    help="normalize input sample weights")
parser.add_argument('-w','--weights', choices=['u','o','f','z'], default='o',
                    help="u: unweighted in training; o: use weights directly from inputs; f: flip all negative weights; z: set all negative weights to zero")
parser.add_argument('--tmva_negweight', choices=['InverseBoostNegWeights','IgnoreNegWeightsInTraining','PairNegWeightsGlobal'], default='IgnoreNegWeightsInTraining',help="TMVA negative weight treatment")

args = parser.parse_args()

###################
# Input files

# signal input file
# ttH nonbb
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

if args.timeit:
    start=timer()
if not args.quiet:
    print 'Reading signal samples ...'

xsig, ysig, wsig = util.read_inputs(infile_sig, var, True)
wsig = util.update_weights(wsig, args.weights)
# scale 
wsig *= 1. * xs_sig / np.sum(wsig)

if not args.quiet:
    print 'number of signal samples: ', xsig.shape[0]
    print 'Reading background sampels ...'

dataset_bkg = []
for in_bkg in infiles_bkg:
    xbi, ybi, wbi = util.read_inputs(in_bkg, var, False)
    wbi = util.update_weights(wbi, args.weights)
    dataset_bkg.append((xbi, ybi, wbi))

# combine background samples
xbkg, ybkg, wbkg = util.combine_inputs(dataset_bkg, xs_bkg, 1.)

if not args.quiet:
    print 'number of background samples: ', xbkg.shape[0]

if args.timeit:
    stop=timer()
    print 'Reading inputs takes ', stop-start, 's'

if args.correlation:
    util.plot_correlation(xsig, var, args.outdir+'correlation_sig.png',
                          verbose=(not args.quiet))
    util.plot_correlation(xbkg, var, args.outdir+'correlation_bkg.png',
                          verbose=(not args.quiet))

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
    
output = TFile(args.outdir+'tmva_output.root', 'recreate')
factory = TMVA.Factory('TMVA', output, 'AnalysisType=Classification:'
                       '!V:Silent:!DrawProgressBar')
dataloader = TMVA.DataLoader("")
TMVA.gConfig().GetIONames().fWeightFileDir = args.outdir+'weights/'

for v in var:
    vtype = 'I' if v in ['nJet','tau0_decaymode','tau1_decaymode','ntags','ntags_loose'] else 'F'
    dataloader.AddVariable(v, vtype)

add_classification_events(dataloader, x_train, y_train, weights=w_train)
add_classification_events(dataloader, x_test, y_test, weights=w_test, test=True)

norm = 'None'
#if args.normalize:
#    norm = 'NumEvents'
dataloader.PrepareTrainingAndTestTree(TCut('1'), 'NormMode={}'.format(norm))

config="NTrees={}:MaxDepth={}:BoostType=Grad:SeparationType=GiniIndex:Shrinkage={}:NegWeightTreatment={}".format(args.ntree,args.depth,args.rate,args.tmva_negweight)

if not args.quiet:
    print config

factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDT", config)
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
    for v in var:
        #vtype = 'i' if v in ['nJet','tau0_decaymode','tau1_decaymode','ntags','ntags_loose'] else 'f'
        
        reader.AddVariable(v, array('f', [0]))

    reader.BookMVA('BDT','dataset/weights/TMVA_BDT.weights.xml')
    y_decision = evaluate_reader(reader, "BDT", x_test)
    util.plot_clf_results_tmva(reader, x_train, y_train, w_train, x_test, y_test, w_test, figname=args.outdir+"bdtoutput.png", verbose=(not args.quiet)) 
    util.plot_roc((y_test, y_decision, w_test),figname=args.outdir+'roc.png')
