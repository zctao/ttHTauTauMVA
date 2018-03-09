#!/usr/bin/env python

import argparse
import ttHTauTauMVA.ttHTauTauMVA.mva_utils as util
import numpy as np
from sklearn.externals import joblib
from root_numpy import array2root, array2tree
from ROOT import TFile, TTree, TObject
from array import array


parser = argparse.ArgumentParser(description='Evaluate BDT with scikit-learn')
parser.add_argument('ntuplefile', type=str, help="Name of ntuple file")
parser.add_argument('bdtname', type=str, help="Name of bdt")
#parser.add_argument('-o', '--output',type=str, help="Output name")
parser.add_argument('-b', '--branchname', type=str, default="mva_score",
                    help="Name of branch")
parser.add_argument('-t','--intreename', type=str, default="mva",
                    help="Name of input tree")
parser.add_argument('-vars', '--variables', nargs='+', type=str,
                    help="List of variables for evaluating. If none provided, take all available branches from input root tree")
parser.add_argument('-w', '--weight', type=str, default='event_weight',
                    help="Name of the event weight branch in ntuple")
parser.add_argument('-q', '--quiet', action='store_true', help="quiet mode")

args = parser.parse_args()

# variables used to evaluate BDT
var = util.get_all_variable_names(args.ntuplefile) if args.variables is None else args.variables
if not args.quiet:
    print "Variables for evaluating: ", var

x, y, w = util.read_inputs(args.ntuplefile, var, True, weight_name=args.weight)

if not args.quiet:
    print "sum of weights (signal) : ", np.sum(w)
    print 'number of events: ', x.shape[0]

# load trained bdt
bdt = joblib.load(args.bdtname)

# bdt scores
y_pred = bdt.predict_proba(x)[:,1]

# add new branch
fin = TFile(args.ntuplefile,'update')
treein = fin.Get(args.intreename)

bdtscore = array('f',[0.])
newbranch = treein.Branch(args.branchname, bdtscore, args.branchname+'/F')

assert(len(y_pred)==treein.GetEntries())

for i in range(treein.GetEntries()):
    treein.GetEntry(i)
    bdtscore[0] = y_pred[i]
    if i==5:
        print y_pred[i]
    newbranch.Fill()
    
treein.Write("", TObject.kOverwrite)
fin.Close()

print "mvaNtuple", args.ntuplefile, "is updated"

#
# convert numpy narray to structured array
#mvascore = np.core.records.fromarrays([y_pred], names=args.branchname)
#
#array2root(mvascore, 'test.root', treename='bdtscore', mode='recreate')
#
#treebdt = array2tree(mvascore,name='bdt')
#
# add new tree as friend to mvaNtuple
#fin = TFile(args.ntuplefile,'update')
#treein = fin.Get(args.intreename)
#
# check its list of friends first
#[f.GetName() for friend in treein.GetListOfFriends()]

#treein.AddFriend(treebdt)
#treein.Write()

