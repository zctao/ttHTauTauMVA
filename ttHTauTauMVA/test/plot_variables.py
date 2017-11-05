import sys
from ROOT import *

if len(sys.argv)!=2:
    exit()

infile = ''
if (sys.argv[1]=='ttH'):
    infile='mvaVars_ttH_loose.root'
elif (sys.argv[1]=='TTW'):
    infile='mvaVars_TTW_loose.root'
elif (sys.argv[1]=='TTZ'):
    infile='mvaVars_TTZ_loose.root'
else:
    print 'usage: python plot_variables.py [sample]'
    print '[sample] = ttH, TTZ or TTW'
    exit()

file = TFile(infile,'read')  
tree = file.Get('mva')
    
variables = """
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

tcanvas = TCanvas()

for var in variables:
    tree.Draw(var, 'event_weight>0', 'norm')
    h_wpos = tree.GetHistogram()
    h_wpos.SetLineColor(2)
    h_wpos.SetTitle(var)

    tree.Draw(var, 'event_weight<0', 'norm same')
    h_wneg = tree.GetHistogram()
    h_wneg.SetLineColor(4)
    h_wneg.SetTitle(var)

    tcanvas.Update()
    tcanvas.SaveAs(var+'_'+sys.argv[1]+'.png')
