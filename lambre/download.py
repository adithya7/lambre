import argparse
import json
import logging
from pathlib import Path
import subprocess
from typing import Dict

URL_REMOTE_PATH = "https://cmu.box.com/shared/static/lsww58ezotrpvnpo7foq9efskxz5nq7c.json"


def download_url_file(dir_path: Path):
    subprocess.run(["wget", "-q", "--show-progress", URL_REMOTE_PATH, "-O", dir_path / "lambre_urls.json"])


def load_resource_paths(dir_path: Path):
    # download file with to latest URLs
    download_url_file(dir_path)
    with open(dir_path / "lambre_urls.json", "r") as rf:
        return json.load(rf)


def parse_args():
    parser = argparse.ArgumentParser(description="download parsers and rules for the input language")
    parser.add_argument("lg", type=str, help="language ISO 639-1 code")
    parser.add_argument(
        "--dir", type=Path, default=Path.home() / "lambre_files", help="path to store lambre related files"
    )

    return parser.parse_args()


def download_lambre_files(lg: str, resource_mapping: Dict, parser_dir: Path, rules_dir: Path):

    logging.info(f"downloading parser and rules for language: {lg}")

    """
    download rules
    """
    # download rules for pratapa-etal-2021
    local_path = rules_dir / "pratapa-etal-2021"
    local_path.mkdir(exist_ok=True, parents=True)
    subprocess.run(
        [
            "wget",
            "-q",
            "--show-progress",
            resource_mapping[lg]["rules_pratapa_etal_2021"],
            "-O",
            local_path / f"{lg}.txt",
        ]
    )
    # download rules for chaudhary-etal-2021
    local_path = rules_dir / "chaudhary-etal-2021"
    local_path.mkdir(exist_ok=True, parents=True)
    subprocess.run(
        [
            "wget",
            "-q",
            "--show-progress",
            resource_mapping[lg]["rules_chaudhary_etal_2021"],
            "-O",
            local_path / f"{lg}.txt",
        ]
    )

    """
    download parser
    """
    # download resources.json
    parser_dir.mkdir(exist_ok=True, parents=True)
    subprocess.run(
        [
            "wget",
            "-q",
            "--show-progress",
            resource_mapping["resources.json"],
            "-O",
            parser_dir / "resources.json",
        ]
    )

    # download parser and decompress
    subprocess.run(
        ["wget", "-q", "--show-progress", resource_mapping[lg]["parser"], "-O", parser_dir / f"{lg}.tar.gz"]
    )
    subprocess.run(["tar", "-xf", parser_dir / f"{lg}.tar.gz", "-C", parser_dir])
    subprocess.run(["rm", parser_dir / f"{lg}.tar.gz"])

    return


def main():

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    args = parse_args()

    args.dir.mkdir(exist_ok=True, parents=True)

    resources_mapping = load_resource_paths(args.dir)

    if args.lg not in resources_mapping:
        logging.warning(f"Language {args.lg} is not supported!")
        logging.warning(
            "Current supported languages: "
            + ", ".join([f'{v["name"]} ({k})' for k, v in resources_mapping.items() if "name" in v])
        )
        exit(1)

    download_lambre_files(
        args.lg, resources_mapping, args.dir / "lambre_stanza_resources", args.dir / "rules"
    )


if __name__ == "__main__":
    main()
