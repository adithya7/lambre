from collections import defaultdict
from copy import deepcopy
from tkinter import E
import numpy as np


def getFeatureValue(feat, feats):
    if feat not in feats:
        return None
    values = list(feats[feat])
    values.sort()
    value = "/".join(values)
    return value


def get_feat_str(feat_dict):
    return "|".join(["%s=%s" % (feat, ",".join(list(feat_dict[feat]))) for feat in feat_dict])


def compute_argstruct_score(argstruct_aggr):
    """ """
    weights = []
    scores = []
    report = {}
    for depd_type in argstruct_aggr:
        for feat in argstruct_aggr[depd_type]:
            for token_type in argstruct_aggr[depd_type][feat]:
                mismatch, match, total = argstruct_aggr[depd_type][feat][token_type]["counts"]
                if mismatch + match > 0 and total > 0:
                    score, weight = float(match) / (match + mismatch), 1
                    weights.append(weight)
                    scores.append(score)
                    report["args=%s:%s:%s" % (depd_type, token_type, feat)] = score

    total_weight = np.sum(weights)
    if total_weight > 0:
        return (
            np.sum([score * weight / total_weight for score, weight in zip(scores, weights)]),
            report,
        )
    # no errors or correct examples found for the pre-specified rules
    return 1.0, {}


def compute_agreement_score(agreement_aggr):
    """
    each agreement type (dep-rel, dim) percentages are computed individually and average is returned
    """
    agr_weights = []
    agr_scores = []
    agr_report = {}
    for agr_type in agreement_aggr:
        for dim in agreement_aggr[agr_type]:
            mismatch, match, total = agreement_aggr[agr_type][dim]
            if mismatch + match > 0:
                agr_score, agr_weight = float(match) / (match + mismatch), 1
                agr_weights.append(agr_weight)
                agr_scores.append(agr_score)
                agr_report["agr=%s:%s" % (agr_type, dim)] = agr_score

    total_weight = np.sum(agr_weights)
    if total_weight > 0:
        return (
            np.sum([score * weight / total_weight for score, weight in zip(agr_scores, agr_weights)]),
            agr_report,
        )
    # no agreements nor disagreements found for the pre-specified rules
    return 1.0, {}


def compute_joint_score(agreement_aggr, argstruct_aggr):

    weights = []
    scores = []
    report = {}
    dim_score_match = defaultdict(lambda: 0)
    dim_score_total = defaultdict(lambda: 0)

    for agr_type in agreement_aggr:
        for dim in agreement_aggr[agr_type]:
            mismatch, match, total = agreement_aggr[agr_type][dim]
            if mismatch + match > 0:
                agr_score, agr_weight = float(match) / (match + mismatch), 1
                weights.append(agr_weight)
                scores.append(agr_score)
                dim_score_match[dim] += match
                dim_score_total[dim] += total
                report["agr=%s:%s" % (agr_type, dim)] = agr_score * 100.0

    for depd_type in argstruct_aggr:
        for feat in argstruct_aggr[depd_type]:
            for token_type in argstruct_aggr[depd_type][feat]:
                mismatch, match, total = argstruct_aggr[depd_type][feat][token_type]["counts"]
                if mismatch + match > 0 and total > 0:
                    score, weight = float(match) / (match + mismatch), 1
                    weights.append(weight)
                    scores.append(score)
                    dim_score_match[feat] += match
                    dim_score_total[feat] += total
                    report["args=%s:%s:%s" % (depd_type, token_type, feat)] = score

    for dim, match in dim_score_match.items():
        total = dim_score_total[dim]
        percentage_match = float(match) / total
        report[f"model: {dim}"] = percentage_match * 100.0

    total_weight = np.sum(weights)
    if total_weight > 0:
        return (
            np.sum([score * weight / total_weight for score, weight in zip(scores, weights)]),
            report,
        )
    else:
        # no errors or correct examples found for the pre-specified rules
        return 1.0, {}


def check_argstruct_rule(token, head_token_idx, argstruct_dict, sent):
    depd_type = "%s-%s-%s" % (
        token.deprel,
        token.upos,
        sent[head_token_idx].upos,
    )
    errors_in_features = []
    errors = []
    if depd_type in argstruct_dict:
        for feat in argstruct_dict[depd_type]:
            token_feat_value = token.feats[feat] if feat in token.feats else None
            head_feat_value = sent[head_token_idx].feats[feat] if feat in sent[head_token_idx].feats else None
            isRuleErrorDepd, isRuleErrorHead = False, False

            depd_dict_value = argstruct_dict[depd_type][feat]["depd"]
            if token_feat_value != None and depd_dict_value["feat_value"] != "-":
                if len(token_feat_value & set(depd_dict_value["feat_value"].split(","))) == 0:
                    isRuleErrorDepd = True

            """ checking for error in argument structure rule w.r.t head """
            head_dict_value = argstruct_dict[depd_type][feat]["head"]
            if head_feat_value != None and head_dict_value["feat_value"] != "-":
                if len(head_feat_value & set(head_dict_value["feat_value"].split(","))) == 0:
                    isRuleErrorHead = True

            """ depd argument structure rule """
            if not isRuleErrorDepd:
                argstruct_dict[depd_type][feat]["depd"]["counts"][1] += 1
            else:
                argstruct_dict[depd_type][feat]["depd"]["counts"][0] += 1
                errors += [(sent, feat, token.id, token_feat_value, head_token_idx, "")]
            if argstruct_dict[depd_type][feat]["depd"]["feat_value"] != "-":
                # only if there is a rule on feat values
                argstruct_dict[depd_type][feat]["depd"]["counts"][2] += 1

            """ head argument structure rule """
            if not isRuleErrorHead:
                argstruct_dict[depd_type][feat]["head"]["counts"][1] += 1
            else:
                argstruct_dict[depd_type][feat]["head"]["counts"][0] += 1
                errors += [(sent, feat, token.id, "", head_token_idx, head_feat_value)]
                errors_in_features.append(feat)

            if argstruct_dict[depd_type][feat]["head"]["feat_value"] != "-":
                # only if there is a rule on feat values
                argstruct_dict[depd_type][feat]["head"]["counts"][2] += 1

    return errors_in_features, errors


def check_agreement(token, head_token_idx, agreement_dict, sent):
    errors = []
    agr_type = "%s-%s-%s" % (
        token.deprel,
        token.upos,
        sent[head_token_idx].upos,
    )
    errors_in_features = []
    if agr_type in agreement_dict:
        for feat in agreement_dict[agr_type]:
            isDisagreement = False
            token_feat_value = token.feats[feat] if feat in token.feats else None
            head_feat_value = sent[head_token_idx].feats[feat] if feat in sent[head_token_idx].feats else None
            if token_feat_value != None and head_feat_value != None:
                if len(token_feat_value & head_feat_value) == 0:
                    isDisagreement = True
                    errors += [(sent, feat, token.id, token_feat_value, head_token_idx, head_feat_value)]

            if not isDisagreement:
                agreement_dict[agr_type][feat][1] += 1
            else:
                agreement_dict[agr_type][feat][0] += 1
                errors_in_features.append(feat)

            agreement_dict[agr_type][feat][2] += 1

    return errors_in_features, errors


def get_sent_score(data, lang_agr, lang_argstruct):
    """
    computes the grammar error metric at sentence level
    """

    scores = []
    sent_examples = []

    for sent in data:
        id2index = sent._ids_to_indexes
        agreement_aggr = {}
        for agr_type in lang_agr:
            agreement_aggr[agr_type] = {}
            for dim_type, _, _ in lang_agr[agr_type]:
                agreement_aggr[agr_type][dim_type] = [0] * 3
        argstruct_aggr = {}
        for depd_type in lang_argstruct:
            argstruct_aggr[depd_type] = {}
            for feat in lang_argstruct[depd_type]:
                depd_feat_value, head_feat_value = lang_argstruct[depd_type][feat]
                argstruct_aggr[depd_type][feat] = {
                    "depd": {
                        "feat_value": depd_feat_value,
                        "counts": [0, 0, 0],
                    },
                    "head": {
                        "feat_value": head_feat_value,
                        "counts": [0, 0, 0],
                    },
                }

        sent_tokens = []
        for token in sent:
            sent_tokens.append(token.form)
        print_examples = []
        for token in sent:
            new_sent_tokens = deepcopy(sent_tokens)
            if token.head != "0" and token.head is not None:
                anns = [token.upos, token.deprel, sent[token.head].upos]
                if not None in anns:
                    errors_in_features, _ = check_agreement(
                        token,
                        token.head,
                        agreement_aggr,
                        sent,
                    )
                    arg_errors_in_features, _ = check_argstruct_rule(
                        token,
                        token.head,
                        argstruct_aggr,
                        sent,
                    )
                    if len(errors_in_features) > 0:
                        for feature in errors_in_features:
                            new_sent_tokens = deepcopy(sent_tokens)
                            token_num = id2index[token.id]
                            token_feature_value = getFeatureValue(feature, token.feats)
                            new_sent_tokens[token_num] = (
                                "***" + new_sent_tokens[token_num] + f"({feature}={token_feature_value})***"
                            )

                            token_head_num = id2index[token.head]
                            token_feature_value = getFeatureValue(feature, sent[token.head].feats)
                            new_sent_tokens[token_head_num] = (
                                "***"
                                + new_sent_tokens[token_head_num]
                                + f"({feature}={token_feature_value})***"
                            )
                            print_examples.append(" ".join(new_sent_tokens) + "\n")
                            print_examples.append(f"{feature} agreement not followed by tokens ***\n")
                            print_examples.append("\n")

                    if len(arg_errors_in_features) > 0:
                        for feature in arg_errors_in_features:
                            new_sent_tokens = deepcopy(sent_tokens)
                            token_num = id2index[token.id]
                            token_feature_value = getFeatureValue(feature, token.feats)
                            new_sent_tokens[token_num] = (
                                "***" + new_sent_tokens[token_num] + f"({feature}={token_feature_value})***"
                            )

                            token_head_num = id2index[token.head]
                            token_feature_value = getFeatureValue(feature, sent[token.head].feats)
                            new_sent_tokens[token_head_num] = (
                                "***"
                                + new_sent_tokens[token_head_num]
                                + f"({feature}={token_feature_value})***"
                            )
                            print_examples.append(" ".join(new_sent_tokens) + "\n")
                            print_examples.append(f"{feature} argstruct not followed by tokens ***\n")
                            print_examples.append("\n")

        agr_score, agr_report = compute_agreement_score(agreement_aggr)
        argstruct_score, argstruct_report = compute_argstruct_score(argstruct_aggr)
        score, report = compute_joint_score(agreement_aggr, argstruct_aggr)
        scores.append(
            {
                "agr": agr_score,
                "agr_report": agr_report,
                "argstruct": argstruct_score,
                "argstruct_report": argstruct_report,
                "joint": score,
                "joint_report": report,
                "sent": " ".join(sent_tokens),
                "examples": "".join(print_examples),
            }
        )

    return scores


def get_doc_score(data, lang_agr, lang_argstruct):
    """
    computes grammar error metric at document level
    """

    """ agreement counts are accumulated for the entire document """
    agreement_aggr = {}
    for agr_type in lang_agr:
        agreement_aggr[agr_type] = {}
        for dim_type in lang_agr[agr_type]:
            agreement_aggr[agr_type][dim_type] = [0] * 3

    argstruct_aggr = {}
    for depd_type in lang_argstruct:
        argstruct_aggr[depd_type] = {}
        for feat in lang_argstruct[depd_type]:
            depd_feat_value, head_feat_value = lang_argstruct[depd_type][feat]
            argstruct_aggr[depd_type][feat] = {
                "depd": {
                    "feat_value": depd_feat_value,
                    "counts": [0, 0, 0],
                },
                "head": {
                    "feat_value": head_feat_value,
                    "counts": [0, 0, 0],
                },
            }

    error_tuples = []
    for sent in data:
        for token in sent:
            if token.head != "0" and token.head is not None:
                anns = [token.upos, token.deprel, sent[token.head].upos]
                if not None in anns:
                    token_error_feats, token_error_tuples = check_agreement(token, token.head, agreement_aggr, sent)
                    error_tuples.extend(token_error_tuples)
                    token_error_feats, token_error_tuples = check_argstruct_rule(
                        token,
                        token.head,
                        argstruct_aggr,
                        sent,
                    )
                    error_tuples.extend(token_error_tuples) 

    score, report = compute_joint_score(agreement_aggr, argstruct_aggr)
    agr_score, agr_report = compute_agreement_score(agreement_aggr)
    argstruct_score, argstruct_report = compute_argstruct_score(argstruct_aggr)

    score_dict = {
        "agr_score": agr_score,
        "agr_report": agr_report,
        "argstruct_score": argstruct_score,
        "argstruct_report": argstruct_report,
        "joint_score": score,
        "joint_report": report,
    }
    return score_dict, error_tuples
