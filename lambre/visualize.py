from pathlib import Path
from typing import List, Tuple

import pyconll
from ipymarkup import format_span_ascii_markup, format_dep_ascii_markup


def visualize_errors(error_tuples: List) -> Tuple[List, List]:
    """
    Visualization of errors using pratapa-etal-2021 rules
    """
    out_spans = []
    out_depds = []

    for sent, feat, token_idx, token_feat_value, head_token_idx, head_feat_value in error_tuples:

        words = [token.form for token in sent]

        """ span anns with POS for dependent and head tokens """
        spans = []
        offset = 0
        for token in sent:
            if token.id in [token_idx, head_token_idx]:
                spans += [(offset, offset + len(token.form), token.upos)]
            offset += len(token.form) + 1
        text = " ".join(words)
        out_spans += [list(format_span_ascii_markup(text, spans))]

        """ depd anns with dependency label and feature values for depd and head tokens """
        depd_anns = [
            (
                int(head_token_idx) - 1,
                int(token_idx) - 1,
                f"{sent[token_idx].deprel} (depd: {feat}={';'.join(list(token_feat_value))}, head: {feat}={';'.join(list(head_feat_value))})",
            )
        ]
        out_depds += [list(format_dep_ascii_markup(words, depd_anns))]

    return out_spans, out_depds


def write_visualizations(file_path: Path, spans: List, depds: List):
    with open(file_path, "w") as wf:
        for span_ann, depd_ann in zip(spans, depds):
            wf.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
            wf.write("----POS tagged sentence----\n\n")
            wf.write("\n".join(span_ann))
            wf.write("\n\n")
            wf.write("----Dependency parse----\n\n")
            wf.write("\n".join(depd_ann))
            wf.write("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


def get_conll_str(sent: pyconll.unit.sentence.Sentence, token_id: int) -> str:
    html_sents = ['<pre><code class="language-conllu">']
    for token in sent:
        if token.id == token_id:
            splits = token.conll().split("\t")
            splits[4] = "_"
            splits[8:] = ["_"] * 2
            html_sents += ["\t".join(splits)]
        elif token.id == sent[token_id].head:
            splits = token.conll().split("\t")
            splits[4] = "_"
            splits[6] = "0"
            splits[7:] = ["_"] * 3
            html_sents += ["\t".join(splits)]
        elif "-" not in token.id:
            splits = ["_"] * 10
            splits[0] = token.id
            splits[1] = token.form
            splits[2] = token.lemma
            splits[3] = token.upos
            splits[6] = "0"
            html_sents += ["\t".join(splits)]
    html_sents += ["\n</code></pre>"]

    return "\n".join(html_sents)


def visualize_conll_errors(error_tuples: List):
    conll_str = []
    for idx, (sent, feat, token_idx, token_feat_value, head_token_idx, head_feat_value) in enumerate(
        error_tuples
    ):
        conll_str += [f"<div class='bibtex' id='{idx}'>"]
        conll_str += [get_conll_str(sent, token_idx)]
        conll_str += ["</div>"]
    return "\n".join(conll_str)


def write_html_visualizations(file_path: Path, conll_examples: str):

    # load header and footer content
    with open("lambre/html_templates/header.html", "r") as rf:
        HEADER = "".join(rf.readlines())
    with open("lambre/html_templates/footer.html", "r") as rf:
        FOOTER = "".join(rf.readlines())

    with open(file_path, "w") as wf:
        wf.write(f"{HEADER}\n")
        wf.write(f"{conll_examples}\n")
        wf.write(f"{FOOTER}\n")
