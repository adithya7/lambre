#!/bin/bash

export UDBASE=$1
export DATA_ROOT=$2
corpus=$3
export SAVE_DIR=$4/$corpus/
mkdir -p $SAVE_DIR

# setting up environment variables
source config.sh

cd stanza
# # tokenize
# python stanza/stanza/utils/datasets/prepare_tokenize_treebank.py $corpus
# python stanza/stanza/utils/training/run_tokenize.py $corpus --model_dir $SAVE_DIR

# # mwt
# python stanza/stanza/utils/datasets/prepare_mwt_treebank.py $corpus
# python stanza/stanza/utils/training/run_mwt.py $corpus --model_dir $SAVE_DIR

# pos
python -m stanza.utils.datasets.prepare_pos_treebank $corpus
python -m stanza.utils.training.run_pos $corpus --save_dir $SAVE_DIR

# lemma
python -m stanza.utils.datasets.prepare_lemma_treebank $corpus
python -m stanza.utils.training.run_lemma $corpus --model_dir $SAVE_DIR

# depparse
python -m stanza.utils.datasets.prepare_depparse_treebank $corpus --gold
python -m stanza.utils.training.run_depparse $corpus --save_dir $SAVE_DIR

cd ..