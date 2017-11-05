#!/bin/bash

# ttV sample
# scikit-learn
# with original weights and normalized
sklearn_train.py ttV -t -e -n -w o --outdir=/uscms/home/ztao/nobackup/BDTs/sklearn/ttV/weighted/
# unweighted in training
sklearn_train.py ttV -t -e -w u --outdir=/uscms/home/ztao/nobackup/BDTs/sklearn/ttV/unweighted/
# flip negative weights
sklearn_train.py ttV -t -e -n -w f --outdir=/uscms/home/ztao/nobackup/BDTs/sklearn/ttV/flipnegweight/
# ignore negative weights
sklearn_train.py ttV -t -e -n -w z --outdir=/uscms/home/ztao/nobackup/BDTs/sklearn/ttV/ignorenegweight/

# tmva
# with original weights and use InverseBoostNegWeights
tmva_train.py ttV -t -e -n -w o --tmva_negweight=InverseBoostNegWeights --outdir=/uscms/home/ztao/nobackup/BDTs/tmva/ttV/weighted/
#unweighted in training
tmva_train.py ttV -t -e -w u --outdir=/uscms/home/ztao/nobackup/BDTs/tmva/ttV/unweighted/
# flip negative weights
tmva_train.py ttV -t -e -n -w f --outdir=/uscms/home/ztao/nobackup/BDTs/tmva/ttV/flipnegweight/
# ignore negative weights
tmva_train.py ttV -t -e -n --tmva_negweight=IgnoreNegWeightsInTraining --outdir=/uscms/home/ztao/nobackup/BDTs/tmva/ttV/ignorenegweight/
