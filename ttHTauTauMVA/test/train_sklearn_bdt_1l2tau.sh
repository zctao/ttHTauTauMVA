#!/bin/bash
era='Dec17'

sklearn_train.py ttV 1l2tau -c -t -e -i ~/nobackup/mvaNtuples/$era/ -o ~/public_html/BDT$era/sklearn/1l2tau/ttV/ -w u -n
sklearn_train.py ttbar 1l2tau -c -t -e -i ~/nobackup/mvaNtuples/$era/ -o ~/public_html/BDT$era/sklearn/1l2tau/ttbar/ -w u -n
