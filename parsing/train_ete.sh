#!/bin/bash

LG=$1
UD_TREEBANK=$2
UD_TREEBANK_SHORT=$3

STANZA_ROOT=$PWD/stanza
ALT_ROOT=$PWD/data/alt-ud-data
ORIG_UD_ROOT=$PWD/data/ud-treebanks-v2.7

DATA=$STANZA_ROOT/raw_data/alt-ud-train-data/
mkdir -p $DATA/$UD_TREEBANK

# using altered train corpus concatenated with original train corpus
cat $ORIG_UD_ROOT/$UD_TREEBANK/${UD_TREEBANK_SHORT}-ud-train.conllu $ALT_ROOT/$UD_TREEBANK/${UD_TREEBANK_SHORT}-ud-train.conllu > $DATA/$UD_TREEBANK/${UD_TREEBANK_SHORT}-ud-train.conllu
cat $ORIG_UD_ROOT/$UD_TREEBANK/${UD_TREEBANK_SHORT}-ud-train.txt $ALT_ROOT/$UD_TREEBANK/${UD_TREEBANK_SHORT}-ud-train.txt > $DATA/$UD_TREEBANK/${UD_TREEBANK_SHORT}-ud-train.txt
# using original dev corpus
cp $ORIG_UD_ROOT/$UD_TREEBANK/${UD_TREEBANK_SHORT}-ud-dev.* $DATA/$UD_TREEBANK/
cp $ORIG_UD_ROOT/$UD_TREEBANK/${UD_TREEBANK_SHORT}-ud-test.* $DATA/$UD_TREEBANK/

EXPT_DATA=$STANZA_ROOT/data/alt-train-ud/$UD_TREEBANK
mkdir -p $EXPT_DATA
rm -rf $EXPT_DATA/*

MODEL_DIR=$STANZA_ROOT/models/alt
mkdir -p $MODEL_DIR

# run stanza training (tagger, lemmatizer and parser)
bash run_stanza_train.sh \
    $DATA \
    $EXPT_DATA \
    $UD_TREEBANK \
    $MODEL_DIR