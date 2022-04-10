import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import List


def load_json(file_path: Path) -> List:
    with open(file_path, "r") as rf:
        return json.load(rf)


def write_json(file_path: Path, data: List):
    with open(file_path, "w") as wf:
        json.dump(data, wf, indent=2)


def rewrite_mapping(in_mapping: List, model_dir: Path) -> List:
    out_mapping = deepcopy(in_mapping)
    for lg in in_mapping:
        if lg in ["url"]:
            continue
        found_lg = False
        if "alias" in in_mapping[lg]:
            file_path = model_dir / in_mapping[lg]["alias"]
            if file_path.is_dir():
                found_lg = True
        else:
            for lg_module in in_mapping[lg]:
                if lg_module in ["default_processors", "default_dependencies", "default_md5", "lang_name"]:
                    continue
                found_package = False
                for lg_package in in_mapping[lg][lg_module]:
                    file_path = model_dir / lg / lg_module / f"{lg_package}.pt"
                    if file_path.is_file():
                        out_mapping[lg][lg_module][lg_package]["md5"] = ""
                        out_mapping[lg]["default_processors"][lg_module] = lg_package
                        if lg_module in out_mapping[lg]["default_dependencies"]:
                            for _item in out_mapping[lg]["default_dependencies"][lg_module]:
                                _item["package"] = lg_package
                        found_package = True
                        found_lg = True
                    else:
                        del out_mapping[lg][lg_module][lg_package]
                if not found_package:
                    del out_mapping[lg][lg_module]
                    if lg_module in out_mapping[lg]["default_processors"]:
                        del out_mapping[lg]["default_processors"][lg_module]
                    if lg_module in out_mapping[lg]["default_dependencies"]:
                        del out_mapping[lg]["default_dependencies"][lg_module]
        if found_lg:
            if "default_md5" in out_mapping[lg]:
                out_mapping[lg]["default_md5"] = ""
        else:
            del out_mapping[lg]

    del out_mapping["url"]

    return out_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare resources.json for stanza models of lambre")
    parser.add_argument("stanza_json", type=Path, help="resources.json")
    parser.add_argument("out_json", type=Path, help="output resources.json")
    parser.add_argument("model_dir", type=Path, help="path to trained stanza-style models")

    args = parser.parse_args()

    resources_mapping = load_json(args.stanza_json)
    out_resource_mapping = rewrite_mapping(resources_mapping, args.model_dir)
    write_json(args.out_json, out_resource_mapping)
