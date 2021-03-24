import pyconll
import os
import sys
from collections import defaultdict
import copy
import numpy as np
from scipy.stats import entropy


def analyze(data):
    # Get necessary counts
    all_counts_shared = {}
    all_counts_agreed = {}
    depd_feat_value_counts = {}
    head_feat_value_counts = {}
    corpus_pos_feat_count = {}
    depd_total_counts = {}
    head_total_counts = {}

    for sentence in data:
        for token in sentence:
            relation = token.deprel
            pos = token.upos
            feats = token.feats

            if pos not in corpus_pos_feat_count:
                corpus_pos_feat_count[pos] = {}
            for f in feats:
                feat_value_str = ",".join(list(feats[f]))
                if f not in corpus_pos_feat_count[pos]:
                    corpus_pos_feat_count[pos][f] = defaultdict(int)
                corpus_pos_feat_count[pos][f][feat_value_str] += 1

            if token.head and token.head != "0":
                head_pos = sentence[token.head].upos
                depd_type = "%s-%s-%s" % (relation, pos, head_pos)
                if depd_type not in all_counts_shared:
                    all_counts_shared[depd_type] = defaultdict(int)
                    all_counts_agreed[depd_type] = defaultdict(int)
                    depd_feat_value_counts[depd_type] = {}
                    head_feat_value_counts[depd_type] = {}
                    depd_total_counts[depd_type] = defaultdict(int)
                    head_total_counts[depd_type] = defaultdict(int)

                head_feats = sentence[token.head].feats

                for f in feats:
                    depd_total_counts[depd_type][f] += 1
                    if f not in depd_feat_value_counts[depd_type]:
                        depd_feat_value_counts[depd_type][f] = defaultdict(int)
                    depd_feat_value_counts[depd_type][f][",".join(list(feats[f]))] += 1
                for f in head_feats:
                    head_total_counts[depd_type][f] += 1
                    if f not in head_feat_value_counts[depd_type]:
                        head_feat_value_counts[depd_type][f] = defaultdict(int)
                    head_feat_value_counts[depd_type][f][
                        ",".join(list(head_feats[f]))
                    ] += 1

    # Get probabilities from counts

    """ corpus pos feat value prob """
    corpus_pos_feat_prob = copy.deepcopy(corpus_pos_feat_count)
    for pos in corpus_pos_feat_count:
        for f in corpus_pos_feat_count[pos]:
            value_sum = np.sum(
                [
                    corpus_pos_feat_count[pos][f][value]
                    for value in corpus_pos_feat_count[pos][f]
                ]
            )
            for value in corpus_pos_feat_count[pos][f]:
                corpus_pos_feat_prob[pos][f][value] /= value_sum

    """ dependent prob """
    depd_feat_value_prob = copy.deepcopy(depd_feat_value_counts)
    for depd_type in depd_feat_value_counts:
        for f in depd_feat_value_counts[depd_type]:
            value_sum = np.sum(
                [
                    depd_feat_value_counts[depd_type][f][value]
                    for value in depd_feat_value_counts[depd_type][f]
                ]
            )
            for value in depd_feat_value_counts[depd_type][f]:
                depd_feat_value_prob[depd_type][f][value] /= value_sum

    """ head prob """
    head_feat_value_prob = copy.deepcopy(head_feat_value_counts)
    for depd_type in head_feat_value_counts:
        for f in head_feat_value_counts[depd_type]:
            value_sum = np.sum(
                [
                    head_feat_value_counts[depd_type][f][value]
                    for value in head_feat_value_counts[depd_type][f]
                ]
            )
            for value in head_feat_value_counts[depd_type][f]:
                head_feat_value_prob[depd_type][f][value] /= value_sum

    return (
        depd_feat_value_prob,
        head_feat_value_prob,
        corpus_pos_feat_prob,
        depd_total_counts,
        head_total_counts,
    )


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
    depd_probs, head_probs, corpus_probs, depd_counts, head_counts = analyze(data)
    for depd_type in depd_probs:
        feats = list(depd_probs[depd_type].keys())
        feats.extend(list(head_probs[depd_type].keys()))
        feats = list(set(feats))

        """ computing relative entropy for each feature of dependent and head in this depd_type """

        _, depd_pos, head_pos = depd_type.split("-")
        depd_rel_entropy = {}
        for feat in corpus_probs[depd_pos]:
            values = corpus_probs[depd_pos][feat].keys()
            global_dist, local_dist = [], []
            if feat in depd_probs[depd_type]:
                for value in values:
                    global_dist.append(corpus_probs[depd_pos][feat][value])
                    if value in depd_probs[depd_type][feat]:
                        local_dist.append(depd_probs[depd_type][feat][value])
                    else:
                        # value is not seen for the feature in this depd. relation
                        local_dist.append(0.0)
            else:
                continue
            depd_rel_entropy[feat] = entropy(local_dist, qk=global_dist)

        head_rel_entropy = {}
        for feat in corpus_probs[head_pos]:
            values = corpus_probs[head_pos][feat].keys()
            global_dist, local_dist = [], []
            if feat in head_probs[depd_type]:
                for value in values:
                    global_dist.append(corpus_probs[head_pos][feat][value])
                    if value in head_probs[depd_type][feat]:
                        local_dist.append(head_probs[depd_type][feat][value])
                    else:
                        # value is not seen for the feature in this depd. relation
                        local_dist.append(0.0)
            else:
                continue
            head_rel_entropy[feat] = entropy(local_dist, qk=global_dist)

        """ printing morphosyntax rules """

        print("## %s %s" % (lang, depd_type))
        for feat in feats:
            depd_count_str, head_count_str = "null", "null"
            depd_total_count = depd_counts[depd_type][feat]
            head_total_count = head_counts[depd_type][feat]
            if feat in depd_probs[depd_type]:
                depd_count_str = "|".join(
                    [
                        "%s=%.3f" % (value, prob)
                        for value, prob in depd_probs[depd_type][feat].items()
                    ]
                )
            if feat in head_probs[depd_type]:
                head_count_str = "|".join(
                    [
                        "%s=%.3f" % (value, prob)
                        for value, prob in head_probs[depd_type][feat].items()
                    ]
                )
            dentropy, hentropy = "-", "-"
            if feat in depd_rel_entropy:
                dentropy = "%.4f" % (depd_rel_entropy[feat])
            if feat in head_rel_entropy:
                hentropy = "%.4f" % (head_rel_entropy[feat])

            print(
                "%s (%d,%d)\t%s (%s)\t%s (%s)"
                % (
                    feat,
                    depd_total_count,
                    head_total_count,
                    depd_count_str,
                    dentropy,
                    head_count_str,
                    hentropy,
                )
            )

    # for pos in corpus_probs:
    #     if len(corpus_probs[pos].keys()) > 0:
    #         print("## %s %s" % (lang, pos))
    #         for feat in corpus_probs[pos]:
    #             count_str = "|".join(
    #                 [
    #                     "%s=%.3f" % (value, prob)
    #                     for value, prob in corpus_probs[pos][feat].items()
    #                 ]
    #             )
    #             print("%s\t%s" % (feat, count_str))
