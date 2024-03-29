import sys
from collections import defaultdict
import numpy as np
from pathlib import Path

INP_FILE = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
OUT_DIR.mkdir(exist_ok=True)
PROB_THRESHOLD = 0.9
COUNT_THRESHOLD_FACTOR = 0.8

lang2agr = {}

with open(INP_FILE, "r") as rf:
    lang = ""
    for line in rf:
        line_txt = line.strip()
        if line_txt.startswith("# lang:"):
            lang = line_txt.split(":")[1]
            lang2agr[lang] = []
        elif not line_txt.startswith("##"):
            rel = line_txt.split("\t")[0]
            dims = line_txt.split("\t")[1:]
            for dim_prob in dims:
                dim, prob, count = dim_prob.split(" ")
                if float(prob) >= PROB_THRESHOLD:
                    lang2agr[lang].append(("%s:%s" % (rel, dim), float(prob), int(count)))

""" count based pruning """
out_lang2agr = {}
for lang in lang2agr:
    out_lang2agr[lang] = defaultdict(list)
    sorted_list = sorted(lang2agr[lang], key=lambda x: x[2], reverse=True)
    total_count = np.sum([x[2] for x in sorted_list])
    curr_count = 0
    for agr, prob, count in sorted_list:
        depd, dim = agr.rsplit(":", 1)
        # out_lang2agr[lang][depd].append("%s|%.2f|%d" % (dim, prob, count))
        out_lang2agr[lang][depd].append(dim)
        curr_count += count
        if curr_count > COUNT_THRESHOLD_FACTOR * total_count:
            break

sorted_langs = sorted(out_lang2agr.keys())
for lang in sorted_langs:
    with open(OUT_DIR / f"{lang}.txt", "w") as wf:
        sorted_depds = sorted(out_lang2agr[lang].keys())
        for depd in sorted_depds:
            sorted_dims = sorted(out_lang2agr[lang][depd])
            for dim in sorted_dims:
                wf.write(f"{lang}\tpratapa-etal-2021\tagreement\t{depd}\t{dim}\n")
