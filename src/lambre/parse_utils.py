"""
generate depd relations
tools: stanza
"""
import logging
from pathlib import Path

import stanza
from stanza.utils.conll import CoNLL


def get_depd_tree(
    doc: str,
    lg: str,
    stanza_model_path: Path,
    tokenize: bool = True,
    ssplit: bool = False,
    cuda: bool = False,
    verbose: bool = False,
) -> str:

    logging.info(f"generating SUD parse for the input document")

    model_dir = str(stanza_model_path)
    if tokenize and ssplit:
        stanza_nlp = stanza.Pipeline(
            lang=lg, dir=model_dir, use_gpu=cuda, verbose=verbose
        )
    elif tokenize:
        stanza_nlp = stanza.Pipeline(
            lang=lg,
            dir=model_dir,
            tokenize_no_ssplit=True,
            use_gpu=cuda,
            verbose=verbose,
        )
    else:
        stanza_nlp = stanza.Pipeline(
            lang=lg,
            dir=model_dir,
            tokenize_pretokenized=True,
            use_gpu=cuda,
            verbose=verbose,
        )

    stanza_doc = stanza_nlp(doc)
    doc_dict = stanza_doc.to_dict()
    conll = CoNLL.convert_dict(doc_dict)
    doc_conll_str = CoNLL.conll_as_string(conll)

    return doc_conll_str
