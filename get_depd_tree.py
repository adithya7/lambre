"""
generate depd relations
tools: stanza
"""
import argparse
import stanza
from stanza.utils.conll import CoNLL


def main(args):
    sents = ""
    with open(args.txt, "r") as rf:
        for line in rf:
            sents += line
            if args.tokenize and not args.ssplit:
                sents += "\n"

    if args.tokenize and args.ssplit:
        stanza_nlp = stanza.Pipeline(lang=args.lg, dir=args.model, use_gpu=args.cuda,)
    elif args.tokenize:
        stanza_nlp = stanza.Pipeline(
            lang=args.lg, dir=args.model, tokenize_no_ssplit=True, use_gpu=args.cuda,
        )
    else:
        stanza_nlp = stanza.Pipeline(
            lang=args.lg, dir=args.model, tokenize_pretokenized=True, use_gpu=args.cuda,
        )

    doc = stanza_nlp(sents)
    doc_dict = doc.to_dict()
    conll = CoNLL.convert_dict(doc_dict)
    doc_conll_str = CoNLL.conll_as_string(conll)
    with open(args.conllu, "w") as wf:
        wf.write(doc_conll_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run stanza")
    parser.add_argument("-txt", type=str, required=True, help="input text file")
    parser.add_argument("-conllu", type=str, required=True, help="output conllu file")
    parser.add_argument("-lg", type=str, required=True, help="input language")
    parser.add_argument("-model", type=str, required=True, help="stanza model dir")
    parser.add_argument("-tokenize", action="store_true", help="to tokenize")
    parser.add_argument("-ssplit", action="store_true", help="to split sentences")
    parser.add_argument("-cuda", action="store_true", help="use gpu if available")

    args = parser.parse_args()

    main(args)
