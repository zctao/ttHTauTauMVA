import sys
from ROOT import *

if len(sys.argv)!=3:
    exit()

sample = sys.argv[1]
indir = sys.argv[2]

infile = indir+"mvaVars_"+sample+"_1l2tau.root"

file = TFile(infile, 'read')
tree = file.Get('mva')

tcanvas = TCanvas()

h2d = TH2F("upsilon",sample,20, -1., 1., 20, -1., 1.)
h2d.GetXaxis().SetTitle("#Upsilon_{#tau+}")
h2d.GetYaxis().SetTitle("#Upsilon_{#tau-}")

for ev in tree:
    h2d.Fill(ev.taup_upsilon,ev.taum_upsilon)

h2d.SetStats(0)  
h2d.Draw("COLZ")
tcanvas.SaveAs("upsilon_"+sample+"_1l2tau.png")
