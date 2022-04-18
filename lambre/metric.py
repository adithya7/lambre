import argparse
import logging
import subprocess
from pathlib import Path

import pyconll

from lambre import RELATION_MAP, RULE_LINKS, rule_utils, score_utils_chaudhary, score_utils_pratapa, visualize
from lambre.parse_utils import get_depd_tree


def parse_args():
    parser = argparse.ArgumentParser(description="compute morphological well-formedness")
    parser.add_argument("lg", type=str, help="input language ISO 639-1 code")
    parser.add_argument("input", type=Path, help="input file (.txt or .conllu)")
    parser.add_argument(
        "--rule-set",
        type=str,
        default="chaudhary-etal-2021",
        help="rule set name (chaudhary-etal-2021 or pratapa-etal-2021)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="out",
        help="specify path to output directory. Stores parser output and error visualizations.",
    )
    parser.add_argument(
        "--ssplit", action="store_true", help="perform sentence segmentation in addition to tokenization"
    )
    parser.add_argument("--report", action="store_true", help="report scores per rule")
    parser.add_argument(
        "--rules-path", type=Path, default=Path.home() / "lambre_files" / "rules", help="path to rule sets"
    )
    parser.add_argument(
        "--stanza-path",
        type=Path,
        default=Path.home() / "lambre_files" / "lambre_stanza_resources",
        help="path to stanza resources",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose output")

    return parser.parse_args()


def main():

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    args = parse_args()

    """
    Check language support
    """
    lang_parser_path = args.stanza_path / args.lg
    if not lang_parser_path.is_dir():
        # try download
        try:
            subprocess.run(["lambre-download", args.lg], check=True)
        except subprocess.CalledProcessError:
            logging.warning(f"skipping scorer")
            exit(1)

    """
    Scorer expects CoNLL-U file with morphological feature values and (SUD) dependency parse 
    """

    args.output.mkdir(exist_ok=True)
    if args.input.suffix == ".conllu":
        # input CoNLL-U file, directly load the file
        sentences = pyconll.load_from_file(args.input)
    else:
        # input txt file, parse
        depd_tree = get_depd_tree(
            txt_path=args.input,
            lg=args.lg,
            stanza_model_path=args.stanza_path,
            ssplit=args.ssplit,
            verbose=args.verbose,
        )
        sentences = pyconll.load_from_string(depd_tree)
        parser_out_path = f"{args.output / f'{args.input.stem}.conllu'}"
        logging.info(f"storing .conllu file at {parser_out_path}")
        with open(parser_out_path, "w") as wf:
            wf.write(depd_tree)

    """
    Load rule sets and score
    """

    rules_file_path = args.rules_path / args.rule_set / f"{args.lg}.txt"

    if not rules_file_path.is_file():
        logging.warning(f"{args.lg} is not supported for rule set {args.rule_set}")
        exit(1)

    # error tuples for visualization
    error_tuples = []

    if args.rule_set == "pratapa-etal-2021":
        lang_agr, lang_argstruct = rule_utils.load_pratapa_etal_2021_rules(rules_file_path)
        doc_score, error_tuples = score_utils_pratapa.get_doc_score(
            sentences, lang_agr, lang_argstruct, verbose=args.verbose
        )

    elif args.rule_set == "chaudhary-etal-2021":
        lang_rules = rule_utils.load_chaudhury_etal_2021_rules(rules_file_path)
        doc_score, error_tuples = score_utils_chaudhary.get_doc_score(
            sentences, lang_rules, verbose=args.verbose
        )
    f = open(args.output / "score.txt", "w")
    logging.info(f"lambre score: {doc_score['joint_score']:.4f}")
    f.write(f"lambre score: {doc_score['joint_score']:.4f}")

    if args.report:
        doc_report = doc_score["joint_report"]
        for rule, score in doc_report.items():
            logging.info(f"{rule}\t{score:.4f}")
            f.write(f"\n{rule}\t{score:.4f}")
    f.close()
    """
    output txt and html visualizations of the grammatical errors
    """
    errors_path = args.output / "errors"
    errors_path.mkdir(exist_ok=True, parents=True)
    logging.info(f"writing grammatical errors to {errors_path}")

    if args.rule_set == "pratapa-etal-2021":
        out_spans, out_depds = visualize.visualize_errors(error_tuples)
        visualize.write_visualizations(errors_path / "errors.txt", out_spans, out_depds)
        out_conll_str = visualize.visualize_conll_errors(error_tuples)
        visualize.write_html_visualizations(errors_path / "errors.html", out_conll_str)
    elif args.rule_set == "chaudhary-etal-2021":
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

        out_spans, out_depds, error_types = visualize.visualize_errors_chau(error_tuples, relation_map)
        visualize.write_visualizations(errors_path / "errors.txt", out_spans, out_depds)
        (
            out_conll_str_agree,
            out_conll_str_wordorder,
            out_conll_str_assignment,
        ) = visualize.visualize_conll_errors_chau(error_tuples, relation_map, rule_links[args.lg])
        if len(out_conll_str_agree) > 0:
            visualize.write_html_visualizations(errors_path / "errors_agreement.html", out_conll_str_agree)
        if len(out_conll_str_wordorder) > 0:
            visualize.write_html_visualizations(
                errors_path / "errors_wordorder.html", out_conll_str_wordorder
            )
        if len(out_conll_str_assignment) > 0:
            visualize.write_html_visualizations(errors_path / "errors_marking.html", out_conll_str_assignment)


if __name__ == "__main__":
    main()
