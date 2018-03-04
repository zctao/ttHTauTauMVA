#!/bin/bash

version='Feb2018'
#version='Feb2018_noGenMatch'
dir=/uscms/home/ztao/nobackup/mvaNtuples/$version/

# ttH
./plot_InputVars.py $dir'mvaVars_ttH_1l2tau.root' -o $dir'plots/ttH/'
# TTZ
./plot_InputVars.py $dir'mvaVars_TTZ_1l2tau.root' -o $dir'plots/TTZ/'
# TTW
./plot_InputVars.py $dir'mvaVars_TTW_1l2tau.root' -o $dir'plots/TTW/'
# TT_DiLep
./plot_InputVars.py $dir'mvaVars_TT_DiLep_1l2tau.root' -o $dir'plots/TT_DiLep/'
# TT_SemiLep
./plot_InputVars.py $dir'mvaVars_TT_SemiLep_1l2tau.root' -o $dir'plots/TT_SemiLep/'
