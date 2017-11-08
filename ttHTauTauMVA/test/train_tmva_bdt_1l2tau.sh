#!/bin/bash
tmva_train.py ttV 1l2tau -c -t -e -i ~/nobackup/mvaNtuples/ -o ~/public_html/BDTNov17/sklearn/1l2tau/ttV/ -w u -n
tmva_train.py ttbar 1l2tau -c -t -e -i ~/nobackup/mvaNtuples/ -o ~/public_html/BDTNov17/sklearn/1l2tau/ttbar/ -w u -n
