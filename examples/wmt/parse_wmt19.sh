#!/bin/bash

hyp_root=examples/wmt/data/wmt19/wmt19-submitted-data-v3/txt/system-outputs/newstest2019
out_root=examples/wmt/data/wmt-parsed/wmt19/wmt19-submitted-data-v3/txt/system-outputs/newstest2019
ref_root=examples/wmt/data/wmt19/wmt19-submitted-data-v3/txt/references

stanza_dir=stanza_resources

for lg in cs de fi ru; do

    hyp_data_dir="${hyp_root}/en-$lg"
    out_dir="${out_root}/en-$lg"
    ref_file="newstest2019-en$lg-ref.$lg"
    mkdir -p $out_dir

    echo "parsing wmt19 system outputs for $lg"

    for sys_file in `ls $hyp_data_dir`; do
        echo $sys_file
        python get_depd_tree.py \
            -txt $hyp_data_dir/$sys_file \
            -conllu $out_dir/$sys_file.stanza.conllu \
            -lg $lg \
            -model $stanza_dir \
            -tokenize
        grew transform \
            -grs tools/converter/grs/UD+_to_SUD.grs \
            -i $out_dir/$sys_file.stanza.conllu \
            -o $out_dir/$sys_file.stanza.SUD.conllu
    done

    echo $ref_file
    python get_depd_tree.py \
        -txt $ref_root/$ref_file \
        -conllu $out_dir/$ref_file.stanza.conllu \
        -lg $lg \
        -model $stanza_dir \
        -tokenize
    grew transform \
        -grs tools/converter/grs/UD+_to_SUD.grs \
        -i $out_dir/$ref_file.stanza.conllu \
        -o $out_dir/$ref_file.stanza.SUD.conllu

done