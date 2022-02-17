from collections import defaultdict
from pathlib import Path
import sys

""" entropy threshold to identify morphosyntactic rules """
# TODO: any principled way to choose this threshold?
ENTROPY_THRESHOLD = 0.9
""" minimum count for each depd link to consider potential morphosyntactic rules """
# TODO: different count thresholds for different treebanks
# MIN_DEPD_COUNT = 100
MIN_DEPD_COUNT = {"ru": 500, "de": 100, "cs": 500, "SUD.de": 500}
""" minimum probability for a feature value to be chosen """
MIN_VALUE_PROB = 0.1

valid_feats = ["Case", "VerbForm"]

INP_FILE = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
OUT_DIR.mkdir(exist_ok=True)

lg2rules = defaultdict(list)
lang, depd_type = "", ""
with open(INP_FILE, "r") as rf:
    rules = []
    for line_raw in rf:
        line = line_raw.strip()
        if line.startswith("# "):
            lang = line.lstrip("# ")
        elif line.startswith("## "):
            for rule in rules:
                depd_val_str, head_val_str = "", ""

                """ extracting possible values of the `feat` for dependent """
                if rule["depd_probs"] != "-":
                    probs = rule["depd_probs"].split("|")
                    for val_prob in probs:
                        val, prob = val_prob.split("=")
                        if float(prob) >= MIN_VALUE_PROB:
                            depd_val_str += val
                            depd_val_str += ";"
                    depd_val_str = depd_val_str.rstrip(";")

                """ extracting possible values of the `feat` for head """
                if rule["head_probs"] != "-":
                    probs = rule["head_probs"].split("|")
                    for val_prob in probs:
                        val, prob = val_prob.split("=")
                        if float(prob) >= MIN_VALUE_PROB:
                            head_val_str += val
                            head_val_str += ";"
                    head_val_str = head_val_str.rstrip(";")

                if depd_val_str != "" or head_val_str != "":
                    if depd_val_str == "":
                        depd_val_str = "-"
                    if head_val_str == "":
                        head_val_str = "-"
                    lg2rules[lang] += [(depd_type, rule["feat"], depd_val_str, head_val_str)]
            lang, depd_type = line.lstrip("## ").split(" ")
            rules = []
        else:
            feat_str, depd_str, head_str = line.split("\t")
            feat = feat_str.split(" ")[0]
            if feat not in valid_feats:
                continue
            depd_count = int(feat_str.split(" ")[1].strip("()").split(",")[0])
            head_count = int(feat_str.split(" ")[1].strip("()").split(",")[1])
            isRule = False

            depd_prob_dist, depd_ent = depd_str.split(" ")
            head_prob_dist, head_ent = head_str.split(" ")
            depd_ent = depd_ent.lstrip("(").rstrip(")")
            head_ent = head_ent.lstrip("(").rstrip(")")
            if depd_ent != "-":
                depd_ent = float(depd_ent)
            else:
                depd_ent = 0.0
            if head_ent != "-":
                head_ent = float(head_ent)
            else:
                head_ent = 0.0

            count_threshold = 100
            if lang in MIN_DEPD_COUNT:
                count_threshold = MIN_DEPD_COUNT[lang]

            if depd_count >= count_threshold and depd_ent >= ENTROPY_THRESHOLD:
                isRule = True
            else:
                # prob distribution is not relevant anymore
                depd_prob_dist = "-"

            if head_count >= count_threshold and head_ent >= ENTROPY_THRESHOLD:
                isRule = True
            else:
                # prob distribution is not relevant anymore
                head_prob_dist = "-"

            if isRule:
                rules.append(
                    {
                        "depd_type": depd_type,
                        "feat": feat,
                        "depd_probs": depd_prob_dist,
                        "head_probs": head_prob_dist,
                    }
                )


for lg in lg2rules:
    with open(OUT_DIR / f"{lg}.txt", "w") as wf:
        for depd_type, feat, depd_val_str, head_val_str in lg2rules[lg]:
            wf.write(
                f"{lg}\tpratapa-etal-2021\targstruct\t{depd_type}\t{feat}\t{depd_val_str}\t{head_val_str}\n"
            )
