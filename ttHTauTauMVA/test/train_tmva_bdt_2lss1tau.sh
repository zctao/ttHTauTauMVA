#!/bin/bash
era='Dec17'

tmva_train.py ttV 2lss1tau -c -t -e -i ~/nobackup/mvaNtuples/$era/ -o ~/public_html/BDT$era/tmva/2lss1tau/ttV/ -w u -n
tmva_train.py ttbar 2lss1tau -c -t -e -i ~/nobackup/mvaNtuples/$era/ -o ~/public_html/BDT$era/tmva/2lss1tau/ttbar/ -w u -n
