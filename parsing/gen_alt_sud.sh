#!/bin/bash

TREEBANK_ROOT=$1
TREEBANKS=$2

UNIMORPH="data/unimorph"
# path to generate noisy (or alternate) treebanks
ALT_DATA='data/alt-sud-data'

mkdir -p $UNIMORPH
mkdir -p $ALT_DATA

while read line || [[ -n $line ]]; do
    read -r lg lg3 treebank short <<<$line

    # download UNIMORPH dictionaries
    if [ ! -f $UNIMORPH/$lg3 ]; then
        echo "downloading UniMorph dictionary for $lg3"
        if [ $lg3 == 'fin' ]; then
            wget -nv -O $UNIMORPH/$lg3 https://raw.githubusercontent.com/unimorph/fin/master/fin.1
        else
            wget -nv -O $UNIMORPH/$lg3 https://raw.githubusercontent.com/unimorph/$lg3/master/$lg3
        fi
    fi

    echo "loading UniMorph dictionary from $UNIMORPH/$lg3"
    echo "generating noisy train data for lg: $lg, lg3: $lg3, treebank: $treebank, treebank_short: $short"
    
    mkdir -p $ALT_DATA/$treebank/
    
    # train
    python sample_noise_ud.py \
        -unimorph $UNIMORPH/$lg3 \
        -orig $TREEBANK_ROOT/$treebank/${short}-sud-train.conllu \
        -alt $ALT_DATA/$treebank/${short}-sud-train.conllu
    # dev
    python sample_noise_ud.py \
        -unimorph $UNIMORPH/$lg3 \
        -orig $TREEBANK_ROOT/$treebank/${short}-sud-dev.conllu \
        -alt $ALT_DATA/$treebank/${short}-sud-dev.conllu
    # test
    python sample_noise_ud.py \
        -unimorph $UNIMORPH/$lg3 \
        -orig $TREEBANK_ROOT/$treebank/${short}-sud-test.conllu \
        -alt $ALT_DATA/$treebank/${short}-sud-test.conllu
        
done < $TREEBANKS