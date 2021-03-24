# Training Robust Dependency Parsers

To train dependency parses that are robust to morphology-related errors in text, we recommend using the UniMorph resource and Stanza training framework.

## Prepare Data

Add morphology-related noise to the train and development splits of the original UD treebanks. We use [Unimorph](https://github.com/unimorph) dictionaries to find alternate inflections for words in UD treebanks. The list of UD treebanks to add noise are listed under `treebanks.txt`.

```bash
TREEBANK_PATH=data/ud-treebanks-v2.7
bash gen_alt.sh \
    $TREEBANK_PATH \
    treebanks.txt
```

## Train

[Stanza](https://stanfordnlp.github.io/stanza/) provides a framework to train dependency parsers for over 70 languages that are available in the UD project. It allows for training tokenizers, lemmatizer, tagger and dependency parser on the treebanks. Note that, sentence segmentation and tokenization modules are not impacted by the morphology-related noise (above), and therefore be reused.

First clone the official stanza repository.

```bash
git clone https://github.com/stanfordnlp/stanza.git
# download word2vec vectors
# (warning) this step can take a long time
bash stanza/scripts/download_vectors.sh stanza_word_vectors
```

To train lemmatizer, tagger and parser, first augment the noisy treebank to the original treebank. Then train robust models as below. We recommend using GPU for training.

```bash
# training for Greek
bash train_ete.sh \
    el \
    UD_Greek-GDT \
    el_gdt
```

## Adding Models and Evaluation

The newly trained models can be used in downstream tasks using the Stanza `Pipeline` tool ([Pipeline and Processors](https://stanfordnlp.github.io/stanza/pipeline.html)). For this, reorganize the models into the required format.

```bash
# download original stanza models
# tokenizers are copied from the original stanza resources
# python -c "import stanza; stanza.download('el', package='gdt', processors='tokenize,mwt', model_dir='original_stanza_resources')"
python -c "import stanza; stanza.download('el', package='gdt', model_dir='original_stanza_resources')"
bash prepare_model_pipeline.sh \
    stanza/models/alt/UD_Greek-GDT \
    original_stanza_resources \
    stanza_resources
```

**Note**: to be able to use the newly trained models using Stanza `Pipeline`, you also need to modify `stanza_resources/resources.json` manually. See [Integrating into Stanza](https://stanfordnlp.github.io/stanza/new_language.html#integrating-into-stanza) for detailed instructions. Refer to `resources_template.json` for a sample.

To evaluate the parsers on original and noisy treebanks, first generate outputs in `conllu` format. Then use the standard evaluation script to get the performance metrics.

```bash
# generate predictions
# morphological noise was added to conllu files, so gold tokenization was assumed

# original dev
python get_depd_tree.py \
    -input data/ud-treebanks-v2.7/UD_Greek-GDT/el_gdt-ud-dev.conllu \
    -output el_gdt-ud-dev.pred.conllu \
    -lg el \
    -model stanza_resources \
    -treebank gdt

# score on original dev
python stanza/stanza/utils/conll18_ud_eval.py -v \
    data/ud-treebanks-v2.7/UD_Greek-GDT/el_gdt-ud-dev.conllu \
    el_gdt-ud-dev.pred.conllu

# original dev (original stanza parsers)
python get_depd_tree.py \
    -input data/ud-treebanks-v2.7/UD_Greek-GDT/el_gdt-ud-dev.conllu \
    -output el_gdt-ud-dev.pred.orig.conllu \
    -lg el \
    -model original_stanza_resources \
    -treebank gdt

# score on original dev (original stanza parsers)
python stanza/stanza/utils/conll18_ud_eval.py -v \
    data/ud-treebanks-v2.7/UD_Greek-GDT/el_gdt-ud-dev.conllu \
    el_gdt-ud-dev.pred.orig.conllu

# alt-dev
python get_depd_tree.py \
    -input data/alt-ud-data/UD_Greek-GDT/el_gdt-ud-dev.conllu \
    -output el_gdt-ud-dev.alt.pred.conllu \
    -lg el \
    -model stanza_resources \
    -treebank gdt

# score on alt-dev
python stanza/stanza/utils/conll18_ud_eval.py -v \
    data/alt-ud-data/UD_Greek-GDT/el_gdt-ud-dev.conllu \
    el_gdt-ud-dev.alt.pred.conllu

# alt-dev (using original stanza parsers)
python get_depd_tree.py \
    -input data/alt-ud-data/UD_Greek-GDT/el_gdt-ud-dev.conllu \
    -output el_gdt-ud-dev.alt.pred.orig.conllu \
    -lg el \
    -model original_stanza_resources \
    -treebank gdt

# score on alt-dev (using original stanza parsers)
python stanza/stanza/utils/conll18_ud_eval.py -v \
    data/alt-ud-data/UD_Greek-GDT/el_gdt-ud-dev.conllu \
    el_gdt-ud-dev.alt.pred.orig.conllu
```

## Parse Text

To parse a raw text corpus using the above trained models

```bash
cd ..
python get_depd_tree.py \
    -txt <TXT_PATH>\
    -conllu <CONLLU_PATH> \
    -lg el \
    -model stanza_resources \
    -tokenize
```
