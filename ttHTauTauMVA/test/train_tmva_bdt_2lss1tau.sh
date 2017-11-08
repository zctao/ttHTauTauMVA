#!/bin/bash
tmva_train.py ttV 2lss1tau -c -t -e -i ~/nobackup/mvaNtuples/ -o ~/public_html/BDTNov17/sklearn/2lss1tau/ttV/ -w u -n
tmva_train.py ttbar 2lss1tau -c -t -e -i ~/nobackup/mvaNtuples/ -o ~/public_html/BDTNov17/sklearn/2lss1tau/ttbar/ -w u -n
