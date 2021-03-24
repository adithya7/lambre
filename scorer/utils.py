"""
utils for metric
"""
from collections import defaultdict


def load_agr_file(LG, AGR_FILE):

    lang2agr = defaultdict(list)
    with open(AGR_FILE, "r") as rf:
        lang = ""
        for line in rf:
            line_txt = line.strip()
            if line_txt.startswith("# lang:"):
                if lang != "":
                    break
                elif LG == line_txt.split(":")[-1]:
                    lang = LG
            elif lang != "":
                for dim in line_txt.split("\t")[1:]:
                    try:
                        dim_type, prob, count = dim.split("|")
                        lang2agr[line_txt.split("\t")[0]].append(
                            (dim_type, prob, count)
                        )
                    except:
                        lang2agr[line_txt.split("\t")[0]].append((dim, None, None))

    return lang2agr


def load_argstruct_file(LG, INP_FILE):

    argstruct_dict = {}
    with open(INP_FILE, "r") as rf:
        for line_str in rf:
            line = line_str.strip()
            line_lg, depd_type, feat, depd_vals, head_vals = line.split("\t")
            if line_lg != LG:
                continue
            if depd_type not in argstruct_dict:
                argstruct_dict[depd_type] = {}
            argstruct_dict[depd_type][feat] = (
                depd_vals.replace(";", ","),
                head_vals.replace(";", ","),
            )

    return argstruct_dict
