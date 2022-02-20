from pathlib import Path
from typing import List, Tuple

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
        text = ' '.join(words)
        out_spans += [list(format_span_ascii_markup(text, spans))]

        """ depd anns with dependency label and feature values for depd and head tokens """
        depd_anns = [(int(head_token_idx)-1, int(token_idx)-1, f"{sent[token_idx].deprel} (depd: {feat}={';'.join(list(token_feat_value))}, head: {feat}={';'.join(list(head_feat_value))})")]
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