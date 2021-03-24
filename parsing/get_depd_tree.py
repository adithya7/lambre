"""
generate dependency parses for inputs in CONLL-U format
"""
import argparse
import stanza
from stanza.utils.conll import CoNLL
import pyconll
import numpy as np


def load_conllu(file_path):
    data = pyconll.load_from_file(file_path)
    sents = []
    for sent_ in data:
        mwt_ids = []
        sent = []
        for token in sent_:
            if "." in token.id:
                # ellipsis can be ignored for creating sentences
                continue
            if "-" in token.id:
                start_id, end_id = token.id.split("-")
                mwt_ids.extend(
                    [str(x) for x in np.arange(int(start_id), int(end_id) + 1)]
                )
            if token.id not in mwt_ids:
                sent.append(token.form)
        sents.append(" ".join(sent))
    return "\n\n".join(sents)


def main(args):

    sents = load_conllu(args.input)

    stanza_nlp = stanza.Pipeline(
        lang=args.lg,
        dir=args.model,
        package=args.treebank,
        tokenize_no_ssplit=True,
        use_gpu=args.cuda,
    )

    doc = stanza_nlp(sents)
    doc_dict = doc.to_dict()
    conll = CoNLL.convert_dict(doc_dict)
    doc_conll_str = CoNLL.conll_as_string(conll)
    with open(args.output, "w") as wf:
        wf.write(doc_conll_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run stanza on CONLL-U files")
    parser.add_argument("-input", type=str, required=True, help="input conllu file")
    parser.add_argument("-output", type=str, required=True, help="output conllu file")
    parser.add_argument("-lg", type=str, required=True, help="input language")
    parser.add_argument("-model", type=str, required=True, help="stanza model dir")
    parser.add_argument("-treebank", type=str, help="treebank")
    parser.add_argument("-cuda", action="store_true", help="use gpu if available")

    args = parser.parse_args()

    main(args)
