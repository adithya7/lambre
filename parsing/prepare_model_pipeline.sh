#!/bin/bash

# copy trained models into the stanza directory format to use with Pipeline
# reuse tokenizer and MWT from original stanza pretrained models

MODEL_DIR=$1
STANZA_PRETRAINED=$2
OUT_DIR=$3

for model in `ls $MODEL_DIR`; do
    
    echo $model
    lg="$(echo $model | awk -F'_' '{print $1}')"

    if [[ "$model" == *pretrain.pt ]]; then
        treebank_shorthand="$(echo $model | awk -F'.' '{print $1}' | awk -F'_' '{print $2}')"
        model_type='pretrain'
        mkdir -p $OUT_DIR/$lg/$model_type
        cp $MODEL_DIR/$model $OUT_DIR/$lg/$model_type/$treebank_shorthand.pt
    else
        treebank_shorthand="$(echo $model | awk -F'_' '{print $2}')"
        model_type="$(echo $model | awk -F'_' '{print $3}' | awk -F'.' '{print $1}')"
        pipeline_type=""

        if [[ "$model_type" == "lemmatizer" ]]; then
            pipeline_type="lemma"
        elif [[ "$model_type" == "tagger" ]]; then
            pipeline_type="pos"
        elif [[ "$model_type" == "parser" ]]; then
            pipeline_type="depparse"
        fi

        if [[ ! -z "$pipeline_type" ]]; then
            mkdir -p $OUT_DIR/$lg/$pipeline_type
            cp $MODEL_DIR/$model $OUT_DIR/$lg/$pipeline_type/$treebank_shorthand.pt
        fi
    fi

    # copy tokenizer and mwt (when available)
    mkdir -p $OUT_DIR/$lg/tokenize/
    cp $STANZA_PRETRAINED/$lg/tokenize/$treebank_shorthand.pt $OUT_DIR/$lg/tokenize/

    if [ -d "$STANZA_PRETRAINED/$lg/mwt" ]; then
        mkdir -p $OUT_DIR/$lg/mwt/
        cp $STANZA_PRETRAINED/$lg/mwt/$treebank_shorthand.pt $OUT_DIR/$lg/mwt/
    fi

done