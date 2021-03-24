#!/bin/bash

LG=$1
TXT=$2

# generate UD-style dependencies
echo "---------------------------"
echo "> Step 1: Generating UD parse"
echo "---------------------------"

# path to downloaded robust stanza models
STANZA_RESOURCES=stanza_resources

# required
# -txt: path to raw text file
# -conllu: path to write parses in CoNLL-U format
# -lg: language code
# -model: path to stanza resources directory. This follows the same directory structure as original Stanza models
# -tokenize: option to tokenize the input if previously untokenized
# -ssplit: option to split the raw text into sentences if previously unsegmented

# to run on GPU, add --cuda

UD_CONLLU='tmp_ud.conllu'
python get_depd_tree.py \
    -txt $TXT \
    -conllu $UD_CONLLU \
    -lg $LG \
    -model ${STANZA_RESOURCES} \
    -tokenize

# Once the UD parses are generated, 
# you can use the UD2SUD converter tool to transform the above UD conllu file to SUD format.

# convert UD to SUD
echo "------------------------------------"
echo "> Step 2: Transforming UD to SUD parse"
echo "------------------------------------"

SUD_CONLLU='tmp_sud.conllu'
grew transform \
    -grs tools/converter/grs/UD+_to_SUD.grs \
    -i $UD_CONLLU \
    -o $SUD_CONLLU

# compute L'AMBRE score
echo "-------------------------"
echo "> Step 3: Computing L'AMBRE"
echo "-------------------------"

# add --report to see performance w.r.t each morpho-syntactic rule
python scorer/metric.py \
    -input $SUD_CONLLU \
    -lg $LG \
    -agr rules/agreement_rules.txt \
    -argstruct rules/argstruct_rules_case_verbform.txt

rm $UD_CONLLU
rm $SUD_CONLLU