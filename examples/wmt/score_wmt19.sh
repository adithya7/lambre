#!/bin/bash

data_dir=examples/wmt/data/wmt-parsed/wmt19/wmt19-submitted-data-v3/txt/system-outputs/newstest2019

for lg in cs de fi ru; do
    echo $lg
    hyp_dir="${data_dir}/en-$lg"
    for sys_file in `ls $hyp_dir/*.SUD.conllu`; do
        echo $sys_file
        python scorer/metric.py \
            -input $sys_file \
            -lg $lg \
            -agr rules/agreement_rules.txt \
            -argstruct rules/argstruct_rules_case_verbform.txt
    done
done