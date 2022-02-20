#!/bin/bash

LG=$1
SUD_TREEBANK=$2
SUD_TREEBANK_SHORT=$3

UD_TREEBANK=`echo $SUD_TREEBANK | sed 's/SUD_/UD_/'`

STANZA_ROOT=$PWD/stanza-1.3.0
ALT_ROOT=$PWD/data/alt-sud-data
ORIG_UD_ROOT=$PWD/data/sud-treebanks-v2.8

DATA=$STANZA_ROOT/raw_data/alt-sud-train-data/
mkdir -p $DATA/$UD_TREEBANK

# using altered train corpus concatenated with original train corpus
cat $ORIG_UD_ROOT/$SUD_TREEBANK/${SUD_TREEBANK_SHORT}-sud-train.conllu $ALT_ROOT/$SUD_TREEBANK/${SUD_TREEBANK_SHORT}-sud-train.conllu > $DATA/$UD_TREEBANK/${SUD_TREEBANK_SHORT}-ud-train.conllu
# using original dev corpus
cp $ORIG_UD_ROOT/$SUD_TREEBANK/${SUD_TREEBANK_SHORT}-sud-dev.conllu $DATA/$UD_TREEBANK/${SUD_TREEBANK_SHORT}-ud-dev.conllu
cp $ORIG_UD_ROOT/$SUD_TREEBANK/${SUD_TREEBANK_SHORT}-sud-test.conllu $DATA/$UD_TREEBANK/${SUD_TREEBANK_SHORT}-ud-test.conllu

EXPT_DATA=$STANZA_ROOT/data/alt-train-sud/$UD_TREEBANK
mkdir -p $EXPT_DATA
rm -rf $EXPT_DATA/*

MODEL_DIR=$STANZA_ROOT/models/alt_sud
mkdir -p $MODEL_DIR

# run stanza training (tagger, lemmatizer and parser)
bash run_stanza_train.sh \
    $DATA \
    $EXPT_DATA \
    $UD_TREEBANK \
    $MODEL_DIR