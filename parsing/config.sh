#!/bin/bash
#
# Set environment variables for the training and testing of stanza modules.

# set earlier
echo $UDBASE
echo $DATA_ROOT

# Set UDBASE to the location of UD data folder
# The data should be CoNLL-U format
# For details, see http://universaldependencies.org/conll18/data.html (CoNLL-18 UD data)
# export UDBASE=./ud-data

# Set NERBASE to the location of NER data folder
# The data should be BIO format
# For details, see https://www.aclweb.org/anthology/W03-0419.pdf (CoNLL-03 NER paper)
# export NERBASE=/path/to/NER

# Set directories to store processed training/evaluation files
# export DATA_ROOT=./data
export TOKENIZE_DATA_DIR=$DATA_ROOT/tokenize
mkdir -p $TOKENIZE_DATA_DIR
export MWT_DATA_DIR=$DATA_ROOT/mwt
mkdir -p $MWT_DATA_DIR
export LEMMA_DATA_DIR=$DATA_ROOT/lemma
mkdir -p $LEMMA_DATA_DIR
export POS_DATA_DIR=$DATA_ROOT/pos
mkdir -p $POS_DATA_DIR
export DEPPARSE_DATA_DIR=$DATA_ROOT/depparse
mkdir -p $DEPPARSE_DATA_DIR
export ETE_DATA_DIR=$DATA_ROOT/ete
mkdir -p $ETE_DATA_DIR
export NER_DATA_DIR=$DATA_ROOT/ner
export CHARLM_DATA_DIR=$DATA_ROOT/charlm

# Set directories to store external word vector data
export WORDVEC_DIR=$PWD/stanza_word_vectors/
