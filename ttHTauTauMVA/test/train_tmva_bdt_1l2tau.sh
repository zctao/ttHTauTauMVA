#!/bin/bash
era='Dec17'

tmva_train.py ttV 1l2tau -c -t -e -i ~/nobackup/mvaNtuples/$era/ -o ~/public_html/BDT$era/tmva/1l2tau/ttV/ -w u -n
tmva_train.py ttbar 1l2tau -c -t -e -i ~/nobackup/mvaNtuples/$era/ -o ~/public_html/BDT$era/tmva/1l2tau/ttbar/ -w u -n
