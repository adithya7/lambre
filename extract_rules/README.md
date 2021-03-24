# Extract Morpho-Syntactic Rules

To extract rules for the SUD treebanks listed in `sud_treebanks.txt`.

```bash
TREEBANK_PATH=../parsing/data/sud-treebanks-v2.7

# agreement rules
python empirical_agreement.py \
    sud_treebanks.txt \
    $TREEBANK_PATH > rules_raw.txt
python prepare_agreement_rules.py \
    rules_raw.txt \
    agreement_rules.txt

# case and verbform assignment rules
python empirical_argstruct.py \
    sud_treebanks.txt \
    $TREEBANK_PATH > rules_raw.txt
python prepare_argstruct_rules.py \
    rules_raw.txt \
    argstruct_rules_case_verbform.txt
```

For human evaluation of rules extracted in selected languages, refer to our paper. To extract rules for new languages/treebanks, add the path to treebank's `.conllu` to the `sud_treebanks.txt` file. For improving the accuracy of L'AMBRE further, you can manually edit the automatically extracted rules. For potential directions, see our analyses on GEI task in our paper.
