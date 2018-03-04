#!/usr/bin/env python

import argparse
import ROOT as r

import ttHTauTauMVA.ttHTauTauMVA.mva_utils as util

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str, help="infile mva ntuple file name")
parser.add_argument('-o','--outdir', type=str, default='./', help="output directory")
parser.add_argument('-t','--treename', type=str, default='mva', help="tree name")

args = parser.parse_args()

if args.infile.split('.')[-1]!='root':
    print "Input is not root file. Abort."
    exit()

file = r.TFile(args.infile, 'read')
tree = file.Get(args.treename)

variables = util.get_all_variable_names(args.infile)
print "variables : ", variables

tcanvas = r.TCanvas()

for var in variables:
    tree.Draw(var)
    tcanvas.SaveAs(args.outdir+var+'.pdf')

# plot upsilon
r.gStyle.SetOptStat(10)
drawVarName='taup_easym:taum_easym'
tree.Draw(drawVarName,'','colz')
tcanvas.SaveAs(args.outdir+'upsilon2D.pdf')

# plot for each decay mode combinations
for dp in ['0','1','10']:
    for dm in ['0','1','10']:
        cuts='taup_decaymode=='+dp+'&&'+'taum_decaymode=='+dm
        tree.Draw(drawVarName,cuts,'colz')
        tcanvas.SaveAs(args.outdir+'upsilon2D_'+dp+'_'+dm+'.pdf')
