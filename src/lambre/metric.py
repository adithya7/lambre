import argparse
import logging
import subprocess
from pathlib import Path
from typing import List

import pyconll

from lambre import (
    RELATION_MAP,
    RULE_LINKS,
    rule_utils,
    score_utils_chaudhary,
    score_utils_pratapa,
    visualize,
)
from lambre.parse_utils import get_depd_tree


def parse_args():
    parser = argparse.ArgumentParser(
        description="compute morphological well-formedness"
    )
    parser.add_argument("lg", type=str, help="input language ISO 639-1 code")
    parser.add_argument("input", type=Path, help="input file (.txt or .conllu)")
    parser.add_argument(
        "--rule-set",
        type=str,
        choices=["chaudhary-etal-2021", "pratapa-etal-2021"],
        default="chaudhary-etal-2021",
        help="rule set name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="out",
        help="specify path to output directory. Stores parser output and error visualizations.",
    )
    parser.add_argument(
        "--score-sent", action="store_true", help="return sentence level scores"
    )
    parser.add_argument(
        "--ssplit",
        action="store_true",
        help="perform sentence segmentation in addition to tokenization",
    )
    parser.add_argument("--report", action="store_true", help="report scores per rule")
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=Path.home() / "lambre_files" / "rules",
        help="path to rule sets",
    )
    parser.add_argument(
        "--stanza-path",
        type=Path,
        default=Path.home() / "lambre_files" / "lambre_stanza_resources",
        help="path to stanza resources",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose output")

    return parser.parse_args()


def check_lang(lg: str, stanza_path: Path) -> bool:
    """Check language support"""
    lang_parser_path = stanza_path / lg
    if not lang_parser_path.is_dir():
        # try download
        try:
            subprocess.run(["lambre-download", lg], check=True)
        except:
            logging.warning(f"skipping scorer")
            return False
    return True


def parse_doc(
    doc: str,
    lg: str,
    stanza_path: Path,
    output: Path,
    ssplit: bool,
    verbose: bool,
    file_name: str = None,
):

    depd_tree = get_depd_tree(
        doc=doc,
        lg=lg,
        stanza_model_path=stanza_path,
        ssplit=ssplit,
        verbose=verbose,
    )
    sentences = pyconll.load_from_string(depd_tree)
    if file_name:
        parser_out_path = output / f"{file_name}.conllu"
        logging.info(f"storing .conllu file at {parser_out_path}")
        with open(parser_out_path, "w") as wf:
            wf.write(depd_tree)

    return sentences


def compute_metric(
    sentences,
    lg: str,
    score_sent: bool,
    rule_set: str,
    rules_path: Path,
    report: bool,
    verbose: bool,
    output: Path,
):

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    """
    Scorer expects CoNLL-U file with morphological feature values and (SUD) dependency parse 
    """

    """
    Load rule sets and score
    """

    rules_file_path = rules_path / rule_set / f"{lg}.txt"

    if not rules_file_path.is_file():
        logging.warning(f"{lg} is not supported for rule set {rule_set}")
        exit(1)

    # error tuples for visualization
    error_tuples = []

    if rule_set == "pratapa-etal-2021":
        lang_agr, lang_argstruct = rule_utils.load_pratapa_etal_2021_rules(
            rules_file_path
        )
        if score_sent:
            sent_scores, error_tuples = score_utils_pratapa.get_sent_score(
                sentences, lang_agr, lang_argstruct, verbose=verbose
            )
        else:
            doc_score, error_tuples = score_utils_pratapa.get_doc_score(
                sentences, lang_agr, lang_argstruct, verbose=verbose
            )

    elif rule_set == "chaudhary-etal-2021":
        lang_rules = rule_utils.load_chaudhury_etal_2021_rules(rules_file_path)
        if score_sent:
            sent_scores, error_tuples = score_utils_chaudhary.get_sent_score(
                sentences, lang_rules, verbose=verbose
            )
        else:
            doc_score, error_tuples = score_utils_chaudhary.get_doc_score(
                sentences, lang_rules, verbose=verbose
            )

    scores_path = output / "score.txt"
    f = open(scores_path, "w")

    # write L'AMBRE scores
    f.write("L'AMBRE scores\n")
    if score_sent:
        logging.info(f"writing sentence-level L'AMBRE scores to {scores_path}")
        for idx, _item in enumerate(sent_scores):
            f.write(
                f"sent_idx: {idx}\tlambre_score: {_item['joint_score']:.4f}\tsent: {_item['sent']}\n"
            )
    else:
        logging.info(f"lambre_score: {doc_score['joint_score']:.4f}")
        f.write(f"lambre_score: {doc_score['joint_score']:.4f}\n")

    # write L'AMBRE scores per rule
    if report:
        logging.info(f"writing sentence-level report to {scores_path}")
        f.write("\nL'AMBRE score per rule\n")
        if score_sent:
            for idx, _item in enumerate(sent_scores):
                f.write(f"\n# sent_idx: {idx}")
                f.write(f"\n# sent: {_item['sent']}")
                for rule, score in _item["joint_report"].items():
                    f.write(f"\n{rule}\t{score:.4f}")

        else:
            doc_report = doc_score["joint_report"]
            for rule, score in doc_report.items():
                f.write(f"\n{rule}\t{score:.4f}")

    f.close()

    """
    output txt and html visualizations of the grammatical errors
    """
    errors_path = output / "errors"
    errors_path.mkdir(exist_ok=True, parents=True)
    logging.info(f"writing grammatical errors to {errors_path}")

    if rule_set == "pratapa-etal-2021":
        out_spans, out_depds = visualize.visualize_errors(error_tuples)
        visualize.write_visualizations(errors_path / "errors.txt", out_spans, out_depds)
        out_conll_str = visualize.visualize_conll_errors(error_tuples)
        visualize.write_html_visualizations(errors_path / "errors.html", out_conll_str)
    elif rule_set == "chaudhary-etal-2021":
        relation_map = {}
        with open(RELATION_MAP, "r") as inp:
            for line in inp.readlines():
                info = line.strip().split(";")
                key = info[0].lower()
                value = info[1]
                relation_map[key] = (value, info[-1])
                if "@x" in key:
                    relation_map[key.split("@x")[0]] = (value, info[-1])
        rule_links = {}
        with open(RULE_LINKS, "r") as inp:
            for line in inp.readlines():
                info = line.strip().split(":")
                rule_links[info[0]] = info[1]

        out_spans, out_depds = visualize.visualize_errors_chau(
            error_tuples, relation_map
        )
        visualize.write_visualizations(errors_path / "errors.txt", out_spans, out_depds)
        (
            out_conll_str_agree,
            out_conll_str_wordorder,
            out_conll_str_assignment,
        ) = visualize.visualize_conll_errors_chau(
            error_tuples, relation_map, rule_links[lg]
        )
        if len(out_conll_str_agree) > 0:
            visualize.write_html_visualizations(
                errors_path / "errors_agreement.html", out_conll_str_agree
            )
        if len(out_conll_str_wordorder) > 0:
            visualize.write_html_visualizations(
                errors_path / "errors_wordorder.html", out_conll_str_wordorder
            )
        if len(out_conll_str_assignment) > 0:
            visualize.write_html_visualizations(
                errors_path / "errors_marking.html", out_conll_str_assignment
            )

    if score_sent:
        return [round(sent_score["joint_score"], 4) for sent_score in sent_scores]
    else:
        return round(doc_score["joint_score"], 4)


def score(
    lg: str,
    doc: List[str],
    rule_set: str = "chaudhary-etal-2021",
    output: Path = "out",
    score_sent: bool = False,
    report: bool = False,
    ssplit: bool = False,
    rules_path: Path = Path.home() / "lambre_files" / "rules",
    stanza_path: Path = Path.home() / "lambre_files" / "lambre_stanza_resources",
    verbose: bool = False,
):
    if not check_lang(lg=lg, stanza_path=stanza_path):
        return

    if ssplit:
        parser_input_doc = "".join(doc)
    else:
        parser_input_doc = "\n\n".join(doc)
    sentences = parse_doc(
        doc=parser_input_doc,
        lg=lg,
        stanza_path=stanza_path,
        output=output,
        ssplit=ssplit,
        verbose=verbose,
    )
    scores = compute_metric(
        sentences=sentences,
        lg=lg,
        score_sent=score_sent,
        rule_set=rule_set,
        rules_path=Path(rules_path),
        report=report,
        verbose=verbose,
        output=Path(output),
    )

    return scores


def main():

    args = vars(parse_args())

    if not check_lang(lg=args["lg"], stanza_path=args["stanza_path"]):
        return

    input = Path(args["input"])
    args["output"].mkdir(exist_ok=True)
    if input.suffix == ".conllu":
        # input CoNLL-U file, directly load the file
        sentences = pyconll.load_from_file(input)
    else:
        # input txt file, parse
        doc = ""
        with open(input, "r") as rf:
            for line in rf:
                doc += line
                if args["tokenize"] and not args["ssplit"]:
                    doc += "\n"
        sentences = parse_doc(
            doc=doc,
            lg=args["lg"],
            stanza_path=args["stanza_path"],
            output=args["output"],
            ssplit=args["ssplit"],
            verbose=args["verbose"],
            file_name=input.stem,
        )

    compute_metric(
        sentences=sentences,
        lg=args["lg"],
        score_sent=args["score_sent"],
        rule_set=args["rule_set"],
        rules_path=args["rules_path"],
        report=args["report"],
        verbose=args["verbose"],
        output=args["output"],
    )


if __name__ == "__main__":
    main()
