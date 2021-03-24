# Morpho-Syntactic Rules

These rules were automatically extracted from Surface-Syntactic Universal Dependency treebanks (v2.5).

* `agreement_rules.txt`: rules governing morphological agreement.
* `argstruct_case_verbform.txt`: rules governing case and verbform assignment.

## Reading the rules

For a detailed understanding of the dependency labels in SUD framework, we refer to the [official documentation](https://surfacesyntacticud.github.io/guidelines/u/).

### Agreement Rules

```
# lang:ru
det-DET-NOUN	Case|0.99|16496	Gender|1.00|11112	Number|0.99|16497
subj-NOUN-VERB	Gender|0.95|7996	Number|0.95|26414
```

Example rules for Russian,

* agreement(*det*, `DET`, `NOUN`): agreement between `DET` and its head `NOUN` on case, gender and number.

* agreement(*subj*, `NOUN`, `VERB`): agreement between `VERB` and its `NOUN` subject on gender and number.

The file also contains statistics for the prevelance of an agreement rule in the corresponding UD treebank. For example,

> det-DET-NOUN Case | 0.99 | 16496

The agreement between `DET` and its head `NOUN` is found in 99\% of the 16496 (*det*, `DET`, `NOUN`) dependency links found in Russian SynTagRus treebank.

### Case and Verbform Assignment

```
ru	subj-NOUN-VERB	Case	Nom	-
ru	comp:obj-NOUN-VERB	Case	Acc	-
```

Example rules for Russian,

* case_assignment(*subj*, `NOUN`, `VERB`): the `NOUN` subject to a `VERB` should be in nominative case.

* case_assignment(*comp:obj*, `NOUN`, `VERB`): the `NOUN` object to a `VERB` should be in accusative case.

## Expanding the Rules

To expand the rule sets in the above languages or to add support for new languages, refer to [Extracting Rules](../extract_rules/README.md).
