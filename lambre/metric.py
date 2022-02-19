import argparse
import logging
from pathlib import Path
import subprocess
import pyconll

from lambre.parse_utils import get_depd_tree
from lambre import rule_utils
from lambre import score_utils_pratapa, score_utils_chaudhury


def parse_args():
    parser = argparse.ArgumentParser(description="compute morphological well-formedness")
    parser.add_argument("lg", type=str, help="input language ISO 639-1 code")
    parser.add_argument("input", type=Path, help="input file (.txt or .conllu)")
    parser.add_argument(
        "--rule-set",
        type=str,
        default="pratapa-etal-2021",
        help="rule set name (chaudhury-etal-2021 or pratapa-etal-2021)",
    )
    parser.add_argument("--conllu", action="store_true", help="expect CoNLL-U input, instead of text")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rules-path", type=Path, default=Path.home() / "lambre_files" / "rules", help="path to rule sets")
    parser.add_argument(
        "--stanza-path", type=Path, default=Path.home() / "lambre_files" / "lambre_stanza_resources", help="path to stanza resources"
    )

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

    if args.conllu:
        # input CoNLL-U file, directly load the file
        sentences = pyconll.load_from_file(args.input)
    else:
        # input txt file, parse
        depd_tree = get_depd_tree(txt_path=args.input, lg=args.lg, stanza_model_path=args.stanza_path)
        sentences = pyconll.load_from_string(depd_tree)

    """
    Load rule sets and score
    """

    rules_file_path = args.rules_path / args.rule_set / f"{args.lg}.txt"

    if not rules_file_path.is_file():
        logging.warning(f"{args.lg} is not supported for rule set {args.rule_set}")
        exit(1)

    if args.rule_set == "pratapa-etal-2021":
        lang_agr, lang_argstruct = rule_utils.load_pratapa_etal_2021_rules(rules_file_path)
        doc_score = score_utils_pratapa.get_doc_score(sentences, lang_agr, lang_argstruct)

    elif args.rule_set == "chaudhury-etal-2021":
        lang_rules = rule_utils.load_chaudhury_etal_2021_rules(rules_file_path)
        doc_score = score_utils_chaudhury.get_doc_score(sentences, lang_rules)

    logging.info(f"score: {doc_score['joint_score']:.4f}")
    if args.report:
        doc_report = doc_score["joint_report"]
        for rule, score in doc_report.items():
            logging.info(f"{rule}\t{score:.4f}")


if __name__ == "__main__":
    main()
