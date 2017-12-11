#!/bin/bash
era='Dec17'

sklearn_train.py ttV 2lss1tau -c -t -e -i ~/nobackup/mvaNtuples/$era/ -o ~/public_html/BDT$era/sklearn/2lss1tau/ttV/ -w u -n
sklearn_train.py ttbar 2lss1tau -c -t -e -i ~/nobackup/mvaNtuples/$era/ -o ~/public_html/BDT$era/sklearn/2lss1tau/ttbar/ -w u -n
