#!/bin/bash

analysis="1l2tau"

variables_tt="mTauTauVis tau2_pt tau1_pt avg_dr_jet dr_lep_tau_lead ptmiss dr_taus mindr_lep_jet mindr_tau1_jet mindr_tau2_jet mT_lep nJet nBjetLoose"

clf_ttbar="/uscms/home/ztao/public_html/BDTFeb2018_noGenMatch/sklearn/1l2tau/alexandra/ttbar/bdt.pkl"

variables_ttV="mTauTauVis dr_taus mT_lep ptmiss avg_dr_jet tau1_pt mindr_tau1_jet dr_lep_tau_sublead lep_conePt tau2_pt mindr_lep_jet costS_tau dr_lep_tau_ss"

clf_ttV="/uscms/home/ztao/public_html/BDTFeb2018_noGenMatch/sklearn/1l2tau/alexandra/ttV/bdt.pkl"

#version=$analysis"_13plus2Vars"
version=$analysis"_13Vars"

mvaNtupleDir=/uscms/home/ztao/nobackup/mvaNtuples/M17/jan2018/

mkdir -p $mvaNtupleDir$version/

timestamp=`date +%Y%m%d`
logfile="$mvaNtupleDir"$version/mvaNtuples_wBDT_"$timestamp".txt

rm $logfile
touch $logfile

for ntuple in $mvaNtupleDir*.root; do
	#echo $ntuple
	name=$(echo $ntuple | rev | cut -d'/' -f1 | rev)
	#echo $name
	ntuple_wbdt=$(echo $name | cut -d'.' -f1)"_wBDT.root"
	#echo $ntuple_wbdt
	ntuple_wbdt=$mvaNtupleDir$version/$ntuple_wbdt
	#echo $ntuple_wbdt
	cp $ntuple $ntuple_wbdt
	
	sklearn_evaluate.py $ntuple_wbdt $clf_ttbar --variables $variables_tt -b mva_tt
	sklearn_evaluate.py $ntuple_wbdt $clf_ttV --variables $variables_ttV -b mva_ttV

	echo $ntuple_wbdt | cat >> "${logfile}"
done

echo "Updated mva ntuples in "$mvaNtupleDir
