import pyconll
import os
import sys
from collections import defaultdict


def find_agreement(feats1, feats2):
    shared = set()
    agreed = set()
    for feat in feats1:
        if feat in feats2:
            shared.add(feat)
            if feats1[feat] == feats2[feat]:
                agreed.add(feat)
    return shared, agreed


def analyze(data):
    # Get necessary counts
    all_counts_shared = {}
    all_counts_agreed = {}
    for sentence in data:
        for token in sentence:
            relation = token.deprel
            pos = token.upos
            feats = token.feats

            if token.head and token.head != "0":
                head_pos = sentence[token.head].upos
                depd_type = "%s-%s-%s" % (relation, pos, head_pos)
                if depd_type not in all_counts_shared:
                    all_counts_shared[depd_type] = defaultdict(int)
                    all_counts_agreed[depd_type] = defaultdict(int)

                head_feats = sentence[token.head].feats
                shared, agreed = find_agreement(feats, head_feats)
                for f in shared:
                    # storing total counts and agreement counts
                    all_counts_shared[depd_type][f] += 1
                    if f in agreed:
                        all_counts_agreed[depd_type][f] += 1

    # Arbitrary -- think about this more
    # print(len(data))
    threshold = 50
    if len(data) < 5000:
        threshold = 20
    if len(data) < 1000:
        threshold = 10
    if len(data) < 200:
        threshold = 5

    # Get probabilities from counts
    prob = defaultdict(list)
    for depd_type in all_counts_shared:
        for f in all_counts_shared[depd_type]:
            if all_counts_shared[depd_type][f] > threshold:
                if f in all_counts_agreed[depd_type]:
                    pr = (
                        all_counts_agreed[depd_type][f]
                        / all_counts_shared[depd_type][f]
                    )
                    prob[depd_type].append((f, pr, all_counts_shared[depd_type][f]))
                else:
                    prob[depd_type].append((f, 0.0, 0))

    for depd_type in prob:
        prob[depd_type].sort(key=lambda tup: -tup[1])
    return prob


# Read the file with the list of the treebanks
TREEBANKS = sys.argv[1]
TREEBANKS_PATH = sys.argv[2]
files = []
with open(TREEBANKS) as rf:
    for line in rf:
        files.append(os.path.join(TREEBANKS_PATH, line.strip()))


d = {}
# ... for each treebank
for f in files:
    lang = f.strip().split("/")[-1].split("_")[0]
    lang_full = f.strip().split("/")[-2].split("-")[0][3:]
    f = f.strip()
    data = pyconll.load_from_file(f"{f}")

    print("# lang:%s" % (lang))
    # Get probabilities and lang specific feat values
    probabilities = analyze(data)
    # Filter out the ones that exhibit agreement
    inter = defaultdict(list)
    for depd_type in probabilities:
        for item in probabilities[depd_type]:
            if item[1] > 0.9:
                inter[depd_type].append(
                    " ".join([item[0], "%.3f" % (item[1]), "%d" % (item[2])])
                )

    for depd_type in inter:
        print("%s\t%s" % (depd_type, "\t".join(inter[depd_type])))
