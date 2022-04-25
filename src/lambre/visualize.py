from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import pyconll
from ipymarkup import format_dep_ascii_markup, format_span_ascii_markup

from lambre import rule_utils


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


def visualize_errors_chau(error_tuples, relation_map) -> Tuple[List, List, List]:
    """
    Visualization of errors using chaudhary-etal-2021 rules
    """
    out_spans = []
    out_depds = []

    for (
        sent,
        token,
        isAgreeError,
        isWordOrderError,
        isAssignmentError,
        agreement_rules_not_followed,
        wordorder_rules_not_followed,
        assignment_rules_not_followed,
    ) in error_tuples:

        # Add the head-dependents
        dep_data_token = defaultdict(list)
        sent_tokens = []

        for token_num, token_ in enumerate(sent):
            dep_data_token[token_.head].append(token_.id)
            sent_tokens.append(token_.form)

        if isAgreeError:
            agreement_examples_per_rules = findWordsWhereAgreementNotFollowed(
                agreement_rules_not_followed, sent, sent_tokens, token, relation_map
            )

        if isWordOrderError:
            wordorder_examples_per_rules = findWordsWhereWordOrderNotFollowed(
                wordorder_rules_not_followed, sent, sent_tokens, token, relation_map
            )
        if isAssignmentError:
            casemarking_examples_per_rules = findWordsWhereMarkingNotFollowed(
                assignment_rules_not_followed, sent, sent_tokens, token, relation_map
            )

        """ span anns with POS for dependent and head tokens """
        spans = []
        offset = 0
        token_idx, head_token_idx = token.id, token.head
        for token in sent:
            if token.id in [token_idx, head_token_idx]:
                spans += [(offset, offset + len(token.form), token.upos)]
            offset += len(token.form) + 1
        text = " ".join(sent_tokens)
        out_spans += [list(format_span_ascii_markup(text, spans))]

        depd_anns = []
        error_types = []
        if isAgreeError:
            for feat, (_, _, _, token_feat_value, head_feat_value) in agreement_examples_per_rules.items():
                """depd anns with dependency label and feature values for depd and head tokens"""
                depd_anns.append(
                    (
                        int(head_token_idx) - 1,
                        int(token_idx) - 1,
                        f"{sent[token_idx].deprel} (depd: {feat}={token_feat_value}, head: {feat}={head_feat_value})",
                    )
                )
                error_types.append(f"agreement-{feat}")
        if isWordOrderError:
            for feat, _ in wordorder_examples_per_rules.items():
                """depd anns with dependency label and feature values for depd and head tokens"""
                depd_anns.append(
                    (
                        int(head_token_idx) - 1,
                        int(token_idx) - 1,
                        f"{sent[token_idx].deprel} (word order violated)",
                    )
                )
                error_types.append(f"wordorder-{feat}")
        if isAssignmentError:
            for feat, (_, _, _, token_feat_value, expected_label) in casemarking_examples_per_rules.items():
                """depd anns with dependency label and feature values for depd and head tokens"""
                depd_anns.append(
                    (
                        int(head_token_idx) - 1,
                        int(token_idx) - 1,
                        f"{sent[token_idx].deprel} (depd: Case={token_feat_value}, expected label: Case={expected_label})",
                    )
                )
                error_types.append(f"assignment-{feat}")

        out_depds += [list(format_dep_ascii_markup(sent_tokens, depd_anns))]

    return out_spans, out_depds, error_types


def write_visualizations(file_path: str, spans: List, depds: List):
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
            splits[1] = f"***{splits[1]}***"
            splits[4] = "_"
            splits[8:] = ["_"] * 2
            html_sents += ["\t".join(splits)]
        elif token.id == sent[token_id].head:
            splits = token.conll().split("\t")
            splits[1] = f"***{splits[1]}***"
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


def visualize_conll_errors_chau(error_tuples, relation_map, lang_id):
    agreement_conll_strs, wordorder_conll_strs, casemarking_conll_strs = [], [], []

    idx = 0
    agreement_conll_strs += [
        f"<h1> The tokens of interest (i.e. have errors according to our rules) are marked in ***, hover over the ***-marked tokens for more grammar information </h1>\n"
        f"<h2> Click on the following links for information on the rules </h2>\n"
    ]
    wordorder_conll_strs += [
        f"<h1> The tokens of interest (i.e. have errors according to our rules) are marked in ***, hover over the ***-marked tokens for more grammar information </h1>\n"
        f"<h2> Click on the following links for information on the rules </h2>\n"
    ]
    casemarking_conll_strs += [
        f"<h1> The tokens of interest (i.e. have errors according to our rules) are marked in ***, hover over the ***-marked tokens for more grammar information </h1>\n"
        f"<h2> Click on the following links for information on the rules </h2>\n"
    ]

    for (
        sent,
        token,
        isAgreeError,
        isWordOrderError,
        isAssignmentError,
        agreement_rules_not_followed,
        wordorder_rules_not_followed,
        assignment_rules_not_followed,
    ) in error_tuples:
        # Add the head-dependents
        dep_data_token = defaultdict(list)
        sent_tokens = []
        error_types = []
        autolex_page = f"https://aditi138.github.io/auto-lex-learn/"
        try:

            for token_num, token_ in enumerate(sent):
                dep_data_token[token_.head].append(token_.id)
                sent_tokens.append(token_.form)

            if isAgreeError:
                agreement_examples_per_rules = findWordsWhereAgreementNotFollowed(
                    agreement_rules_not_followed, sent, sent_tokens, token, relation_map
                )
                erro_feats = list(agreement_examples_per_rules.keys())

                agreement_conll_strs += [
                    f'<h3> Error in morphological agreement for <b> {", ".join(erro_feats)} </b> i.e. the ***-marked tokens should have matching gender values </h3>\n'
                ]
                agreement_conll_strs += [f"<table>\n"]
                for erro_feat in erro_feats:
                    prop = erro_feat.split("-")[0].title()
                    pos = erro_feat.split("-")[1]
                    rule_page = f"{autolex_page}/{lang_id}/Agreement/{prop}/{pos}/{pos}.html"
                    agreement_conll_strs += [
                        f'<tr> {prop} agreement in {pos} </tr> <tr> <a href="{rule_page}">Click here</a></tr>\n'
                    ]
                    error_types.append(f"agreement-{erro_feat}")
                agreement_conll_strs += [f"</table>"]
                agreement_conll_strs += [f"<div class='bibtex' id='{idx}'>"]
                agreement_conll_strs += [get_conll_str(sent, token.id)]
                agreement_conll_strs += ["</div>"]

            if isWordOrderError:
                wordorder_examples_per_rules = findWordsWhereWordOrderNotFollowed(
                    wordorder_rules_not_followed, sent, sent_tokens, token, relation_map
                )
                erro_feats = list(wordorder_examples_per_rules.keys())
                wordorder_conll_strs += [
                    f'<h3> Error in word order for <b> {", ".join(erro_feats)} </b> i.e. the token and its syntactic head are not in the correct order </h3>'
                ]
                wordorder_conll_strs += [f"<table>\n"]

                for erro_feat in erro_feats:
                    rule_page = f"{autolex_page}/{lang_id}/WordOrder/{erro_feat}/{erro_feat}.html"
                    wordorder_conll_strs += [
                        f'<tr> word order for {erro_feat} </tr> <tr> <a href="{rule_page}">Click here</a></tr>\n'
                    ]
                    error_types.append(f"wordorder-{erro_feat}")
                wordorder_conll_strs += [f"</table>"]
                wordorder_conll_strs += [f"<div class='bibtex' id='{idx}'>"]
                wordorder_conll_strs += [get_conll_str(sent, token.id)]
                wordorder_conll_strs += ["</div>"]

            if isAssignmentError:
                casemarking_examples_per_rules = findWordsWhereMarkingNotFollowed(
                    assignment_rules_not_followed, sent, sent_tokens, token, relation_map
                )
                erro_feats = list(casemarking_examples_per_rules.keys())
                casemarking_conll_strs += [
                    f'<h3> Error in case marking for <b> {", ".join(erro_feats)} </b> i.e. the case value for *** marked token is not correct </h3>'
                ]
                casemarking_conll_strs += [f"<table>\n"]

                for erro_feat in erro_feats:
                    rule_page = f"{autolex_page}/{lang_id}/CaseMarking/{erro_feat}/{erro_feat}.html"
                    casemarking_conll_strs += [
                        f'<tr> case marking for {erro_feat} </tr> <tr> <a href="{rule_page}">Click here</a></tr>\n'
                    ]
                    error_types.append(f"assignment-{erro_feat}")
                casemarking_conll_strs += [f"</table>"]
                casemarking_conll_strs += [f"<div class='bibtex' id='{idx}'>"]
                casemarking_conll_strs += [get_conll_str(sent, token.id)]
                casemarking_conll_strs += ["</div>"]
        except Exception as e:
            continue
        idx += 1
    return "\n".join(agreement_conll_strs), "\n".join(wordorder_conll_strs), "\n".join(casemarking_conll_strs)


def write_html_visualizations(file_path: Path, conll_examples: str):

    # load header and footer content
    with open(f"{Path(__file__).parent.resolve()}/html_templates/header.html", "r") as rf:
        HEADER = "".join(rf.readlines())
    with open(f"{Path(__file__).parent.resolve()}/html_templates/footer.html", "r") as rf:
        FOOTER = "".join(rf.readlines())

    with open(file_path, "w") as wf:
        wf.write(f"{HEADER}\n")
        wf.write(f"{conll_examples}\n")
        wf.write(f"{FOOTER}\n")


def findWordsWhereAgreementNotFollowed(rules_not_followed, sent, sent_tokens, token, relation_map):
    id2index = sent._ids_to_indexes
    token_num = id2index[token.id]
    token = sent[token_num]
    rules_per_features = {}

    for model, info in rules_not_followed.items():
        if len(info) == 0:
            continue
        for (one_active, one_nonactive, label) in info:
            sent_example_tokens = deepcopy(sent_tokens)
            token_feature_value = rule_utils.getFeatureValue(model, token.feats)
            sent_example_tokens[token_num] = (
                "***" + sent_example_tokens[token_num] + f"({model}={token_feature_value})***"
            )

            token_head_num = id2index[token.head]
            headtoken_feature_value = rule_utils.getFeatureValue(model, sent[token.head].feats)
            sent_example_tokens[token_head_num] = (
                "***" + sent_example_tokens[token_head_num] + f"({model}={headtoken_feature_value})***"
            )

            # sent_error_examples.append(f'Example: {" ".join(sent_example_tokens)}\n')
            # sent_error_examples.append(
            #     f"{model} agreement not followed by tokens marked *** because following rule was not satisfied:\n"
            # )
            # Readable active features
            active_text, nonactive_text = getActiveFeatures(
                one_active, one_nonactive, "agreement", model, relation_map
            )
            rules_per_features[model] = (
                sent_example_tokens,
                active_text,
                nonactive_text,
                token_feature_value,
                headtoken_feature_value,
            )

    return rules_per_features


def findWordsWhereWordOrderNotFollowed(rules_not_followed, sent, sent_tokens, token, relation_map):
    id2index = sent._ids_to_indexes
    token_num = id2index[token.id]
    token = sent[token_num]
    rules_per_features = {}

    for model, info in rules_not_followed.items():
        if len(info) == 0:
            continue
        for (one_active, one_nonactive, label) in info:
            dep, head = model.split("-")[0], model.split("-")[1]
            if head == "noun":
                head = "nominal"
            if label == "before":
                not_label = "after"
            else:
                not_label = "before"

            sent_example_tokens = deepcopy(sent_tokens)
            sent_example_tokens[token_num] = "***" + sent_example_tokens[token_num] + f"({dep})***"

            token_head_num = id2index[token.head]
            sent_example_tokens[token_head_num] = "***" + sent_example_tokens[token_head_num] + f"({head})***"

            # sent_error_examples.append(f'Example: {" ".join(sent_example_tokens)}\n')
            # sent_error_examples.append(
            #     f"{model} order not followed for tokens marked ***, predicted order is {label} but observed is {not_label}. because following rule was not satisfied:\n"
            # )
            # Readable active features
            active_text, nonactive_text = getActiveFeatures(
                one_active, one_nonactive, "wordorder", model, relation_map
            )
            rules_per_features[model] = (sent_example_tokens, active_text, nonactive_text, None, None)
    return rules_per_features


def findWordsWhereMarkingNotFollowed(rules_not_followed, sent, sent_tokens, token, relation_map):
    id2index = sent._ids_to_indexes
    token_num = id2index[token.id]
    token = sent[token_num]
    rules_per_features = {}

    for model, info in rules_not_followed.items():
        if len(info) == 0:
            continue
        for (one_active, one_nonactive, label) in info:
            sent_example_tokens = deepcopy(sent_tokens)
            value = rule_utils.getFeatureValue("Case", token.feats)
            sent_example_tokens[token_num] = (
                "***" + sent_example_tokens[token_num] + f"({token.upos}'s Case={value})***"
            )

            # sent_error_examples.append(f'Example: {" ".join(sent_example_tokens)}\n')
            # sent_error_examples.append(
            #     f"{model} agreement not followed by tokens marked *** because following rule was not satisfied:\n"
            # )
            # Readable active features
            active_text, nonactive_text = getActiveFeatures(
                one_active, one_nonactive, "assignment", model, relation_map
            )
            rules_per_features[model] = (sent_example_tokens, active_text, nonactive_text, value, label)

    return rules_per_features


def getActiveFeatures(active, non_active, task, model, relation_map):
    active_text, nonactive_text = "Required Active features in the rule:\n", "Not Active:"
    if len(active) > 0:
        covered_act = set()
        for a in active:
            if a in covered_act:
                continue
            active_human = rule_utils.transformRulesIntoReadable(a, task, model, relation_map)
            covered_act.add(a)
            active_text += active_human + "\n"

    if len(non_active) > 0:
        covered_na = set()
        for n in non_active:
            if n in covered_na or n:
                continue
            nonactive_human = rule_utils.transformRulesIntoReadable(n, task, model, relation_map)
            covered_na.add(n)
            nonactive_text += nonactive_human + "\n"

    return active_text, nonactive_text
