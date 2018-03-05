#!/bin/bash
#era='Feb2018_noGenMatch'
era='Feb2018'

outdir=/uscms/home/ztao/public_html/BDT$era/sklearn/1l2tau/

ntree=1000
rate=0.02
depth=3

# variables used to train against ttbar background
# Matthias variables used in 2016 ttH,H->tautau analysis
vars_ttbar_matthias="
mTauTauVis
dr_taus
tau1_pt
tau2_pt
avg_dr_jet
ht
nJet
nBjetLoose
"
mkdir -p $outdir'matthias/ttbar/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_matthias -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'matthias/ttbar/'

# Alexandra's variables
# against ttbar
vars_ttbar_xanda="
mTauTauVis
tau2_pt
tau1_pt
avg_dr_jet
dr_lep_tau_lead
ptmiss
dr_taus
mindr_lep_jet
mindr_tau1_jet
mindr_tau2_jet
mT_lep
nJet
nBjetLoose
"
# against ttV
vars_ttV_xanda="
mTauTauVis
dr_taus
mT_lep
ptmiss
avg_dr_jet
tau1_pt
mindr_tau1_jet
dr_lep_tau_sublead
lep_conePt
tau2_pt
mindr_lep_jet
costS_tau
dr_lep_tau_ss
"

mkdir -p $outdir'alexandra/ttbar/'
mkdir -p $outdir'alexandra/ttV/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_xanda -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'alexandra/ttbar/'
#sklearn_train.py ttV 1l2tau $ntree $rate $depth --variables $vars_ttV_xanda -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'alexandra/ttV/'

# Extra variables: tau helicity correlation
# Matthias' + extra
vars_ttbar_m_extra5vars="
mTauTauVis
dr_taus
tau1_pt
tau2_pt
avg_dr_jet
ht
nJet
nBjetLoose
taup_decaymode
taum_decaymode
taup_easym
taum_easym
evisTausAsym
"

mkdir -p $outdir'm_extra5vars/ttbar/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_m_extra5vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'm_extra5vars/ttbar/'

# Alexandra's + extra
vars_ttbar_a_extra5vars="
mTauTauVis
tau2_pt
tau1_pt
avg_dr_jet
dr_lep_tau_lead
ptmiss
dr_taus
mindr_lep_jet
mindr_tau1_jet
mindr_tau2_jet
mT_lep
nJet
nBjetLoose
taup_decaymode
taum_decaymode
taup_easym
taum_easym
evisTausAsym
"
vars_ttV_a_extra5vars="
mTauTauVis
dr_taus
mT_lep
ptmiss
avg_dr_jet
tau1_pt
mindr_tau1_jet
dr_lep_tau_sublead
lep_conePt
tau2_pt
mindr_lep_jet
costS_tau
dr_lep_tau_ss
taup_decaymode
taum_decaymode
taup_easym
taum_easym
evisTausAsym
"

mkdir -p $outdir'a_extra5vars/ttbar/'
mkdir -p $outdir'a_extra5vars/ttV/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_a_extra5vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'a_extra5vars/ttbar/'
#sklearn_train.py ttV 1l2tau $ntree $rate $depth --variables $vars_ttV_a_extra5vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'a_extra5vars/ttV/'


## Only add taup_easym and taum_easym
vars_ttbar_m_extra2vars="
mTauTauVis
dr_taus
tau1_pt
tau2_pt
avg_dr_jet
ht
nJet
nBjetLoose
taup_easym
taum_easym
"

mkdir -p $outdir'm_extra2vars/ttbar/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_m_extra2vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'm_extra2vars/ttbar/'

# Alexandra's + extra
vars_ttbar_a_extra2vars="
mTauTauVis
tau2_pt
tau1_pt
avg_dr_jet
dr_lep_tau_lead
ptmiss
dr_taus
mindr_lep_jet
mindr_tau1_jet
mindr_tau2_jet
mT_lep
nJet
nBjetLoose
taup_easym
taum_easym
"
vars_ttV_a_extra2vars="
mTauTauVis
dr_taus
mT_lep
ptmiss
avg_dr_jet
tau1_pt
mindr_tau1_jet
dr_lep_tau_sublead
lep_conePt
tau2_pt
mindr_lep_jet
costS_tau
dr_lep_tau_ss
taup_easym
taum_easym
"

mkdir -p $outdir'a_extra2vars/ttbar/'
mkdir -p $outdir'a_extra2vars/ttV/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_a_extra2vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'a_extra2vars/ttbar/'
#sklearn_train.py ttV 1l2tau $ntree $rate $depth --variables $vars_ttV_a_extra2vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'a_extra2vars/ttV/'

# Add only evisTausAsym
vars_ttbar_m_evisTausAsym="
mTauTauVis
dr_taus
tau1_pt
tau2_pt
avg_dr_jet
ht
nJet
nBjetLoose
evisTausAsym
"

mkdir -p $outdir'm_evisTausAsym/ttbar/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_m_evisTausAsym -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'm_evisTausAsym/ttbar/'


vars_ttbar_a_evisTausAsym="
mTauTauVis
tau2_pt
tau1_pt
avg_dr_jet
dr_lep_tau_lead
ptmiss
dr_taus
mindr_lep_jet
mindr_tau1_jet
mindr_tau2_jet
mT_lep
nJet
nBjetLoose
evisTausAsym
"
vars_ttV_a_evisTausAsym="
mTauTauVis
dr_taus
mT_lep
ptmiss
avg_dr_jet
tau1_pt
mindr_tau1_jet
dr_lep_tau_sublead
lep_conePt
tau2_pt
mindr_lep_jet
costS_tau
dr_lep_tau_ss
evisTausAsym
"

mkdir -p $outdir'a_evisTausAsym/ttbar/'
mkdir -p $outdir'a_evisTausAsym/ttV/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_a_evisTausAsym -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'a_evisTausAsym/ttbar/'
#sklearn_train.py ttV 1l2tau $ntree $rate $depth --variables $vars_ttV_a_evisTausAsym -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'a_evisTausAsym/ttV/'


# Replace two variables
vars_ttbar_m_repl2vars="
mTauTauVis
dr_taus
tau1_pt
tau2_pt
avg_dr_jet
ht
taup_easym
taum_easym
"

mkdir -p $outdir'm_repl2vars/ttbar/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_m_repl2vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'm_repl2vars/ttbar/'


vars_ttbar_a_repl2vars="
mTauTauVis
tau2_pt
tau1_pt
avg_dr_jet
dr_lep_tau_lead
ptmiss
dr_taus
mindr_lep_jet
mindr_tau1_jet
mindr_tau2_jet
mT_lep
taup_easym
taum_easym
"
vars_ttV_a_repl2vars="
mTauTauVis
dr_taus
mT_lep
ptmiss
avg_dr_jet
tau1_pt
mindr_tau1_jet
dr_lep_tau_sublead
lep_conePt
mindr_lep_jet
dr_lep_tau_ss
taup_easym
taum_easym
"

mkdir -p $outdir'a_repl2vars/ttbar/'
mkdir -p $outdir'a_repl2vars/ttV/'

#sklearn_train.py ttbar 1l2tau $ntree $rate $depth --variables $vars_ttbar_a_repl2vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'a_repl2vars/ttbar/'
#sklearn_train.py ttV 1l2tau $ntree $rate $depth --variables $vars_ttV_a_repl2vars -c -t -e -n -wn xsection_weight -i ~/nobackup/mvaNtuples/$era/ -o $outdir'a_repl2vars/ttV/'
