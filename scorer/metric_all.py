import pyconll
import argparse
import numpy as np
import utils
from collections import defaultdict
from copy import deepcopy

def get_feat_str(feat_dict):
    return "|".join(
        ["%s=%s" % (feat, ",".join(list(feat_dict[feat]))) for feat in feat_dict]
    )


def compute_argstruct_score(argstruct_aggr):
    """

    """
    weights = []
    scores = []
    report = {}
    for depd_type in argstruct_aggr:
        for feat in argstruct_aggr[depd_type]:
            for token_type in argstruct_aggr[depd_type][feat]:
                mismatch, match, total = argstruct_aggr[depd_type][feat][token_type][
                    "counts"
                ]
                if mismatch + match > 0 and total > 0:
                    score, weight = float(match) / (match + mismatch), 1
                    weights.append(weight)
                    scores.append(score)
                    report["args=%s:%s:%s" % (depd_type, token_type, feat)] = score

    total_weight = np.sum(weights)
    if total_weight > 0:
        return (
            np.sum(
                [
                    score * weight / total_weight
                    for score, weight in zip(scores, weights)
                ]
            ),
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
            np.sum(
                [
                    score * weight / total_weight
                    for score, weight in zip(agr_scores, agr_weights)
                ]
            ),
            agr_report,
        )
    # no agreements nor disagreements found for the pre-specified rules
    return 1.0, {}


def compute_joint_score(agreement_aggr, wordorder_aggr, assignment_aggr, argstruct_aggr):

    weights = []
    scores = []
    report = {}

    dim_score_match = defaultdict(lambda : 0)
    dim_score_total = defaultdict(lambda : 0)

    for agr_type in agreement_aggr:
        for dim in agreement_aggr[agr_type]:
            mismatch, match, total = agreement_aggr[agr_type][dim]
            if mismatch + match > 0:
                agr_score, agr_weight = float(match) / (match + mismatch), 1
                weights.append(agr_weight)
                scores.append(agr_score)
                dim_score_match[dim] += match
                dim_score_total[dim] += total
                report["agr = %s:%s" % (agr_type, dim)] = agr_score * 100.0

    for dim in wordorder_aggr:
        mismatch, match, total = wordorder_aggr[dim]
        if mismatch + match > 0:
            agr_score, agr_weight = float(match) / (match + mismatch), 1
            weights.append(agr_weight)
            scores.append(agr_score)
            dim_score_match[dim] += match
            dim_score_total[dim] += total
            report["wo = %s" % (dim)] = agr_score * 100.0

    for dim in assignment_aggr:
        mismatch, match, total = assignment_aggr[dim]
        if mismatch + match > 0:
            agr_score, agr_weight = float(match) / (match + mismatch), 1
            weights.append(agr_weight)
            scores.append(agr_score)
            dim_score_match[dim] += match
            dim_score_total[dim] += total
            report["assignment = %s" % (dim)] = agr_score * 100.0

    for depd_type in argstruct_aggr:# we only do for Case
        for token_type in argstruct_aggr[depd_type]:
            mismatch, match, total = argstruct_aggr[depd_type][token_type][
                "counts"
            ]
            if mismatch + match > 0 and total > 0:
                score, weight = float(match) / (match + mismatch), 1
                weights.append(weight)
                scores.append(score)
                report["args = %s:%s:%s" % (depd_type, token_type, 'Case')] = score  * 100.0

    for dim, match in dim_score_match.items():
        total = dim_score_total[dim]
        percentage_match = float(match) / total
        report[f'model: {dim}'] = percentage_match * 100.0

    total_weight = np.sum(weights)
    if total_weight > 0:
        return (
            np.sum(
                [
                    score * weight / total_weight
                    for score, weight in zip(scores, weights)
                ]
            ),
            report,
        )
    else:
        # no errors or correct examples found for the pre-specified rules
        return 1.0, {}


def compute_score(aggr, task, argstruct_aggr):
    weights = []
    scores = []
    agr_report = {}

    dim_score_match = defaultdict(lambda: 0)
    dim_score_total = defaultdict(lambda: 0)

    if task == 'agreement':
        for agr_type in aggr:
            for dim in aggr[agr_type]:
                mismatch, match, total = aggr[agr_type][dim]
                if mismatch + match > 0:
                    agr_score, agr_weight = float(match) / (match + mismatch), 1
                    weights.append(agr_weight)
                    scores.append(agr_score)
                    dim_score_match[dim] += match
                    dim_score_total[dim] += total
                    agr_report["agr = %s:%s" % (agr_type, dim)] = agr_score * 100.0
    elif task == 'wordorder':
        for dim in aggr:
            mismatch, match, total = aggr[dim]
            if mismatch + match > 0:
                agr_score, agr_weight = float(match) / (match + mismatch), 1
                weights.append(agr_weight)
                scores.append(agr_score)
                dim_score_match[dim] += match
                dim_score_total[dim] += total
                agr_report["wordorder = %s" % (dim)] = agr_score * 100.0
    elif task == 'casemarking':
        for dim in aggr:
            mismatch, match, total = aggr[dim]
            if mismatch + match > 0:
                agr_score, agr_weight = float(match) / (match + mismatch), 1
                weights.append(agr_weight)
                scores.append(agr_score)
                dim_score_match[dim] += match
                dim_score_total[dim] += total
                agr_report["assignment = %s" % (dim)] = agr_score * 100.0

        for depd_type in argstruct_aggr:  # we only do for Case
            for token_type in argstruct_aggr[depd_type]:
                mismatch, match, total = argstruct_aggr[depd_type][token_type][
                    "counts"
                ]
                if mismatch + match > 0 and total > 0:
                    score, weight = float(match) / (match + mismatch), 1
                    # weights.append(weight)
                    # scores.append(score)
                    agr_report["args=%s:%s:%s" % (depd_type, token_type, 'Case')] = score * 100.0

    total_weight = np.sum(weights)
    if total_weight > 0:
        return (
            np.sum(
                [
                    score * weight / total_weight
                    for score, weight in zip(scores, weights)
                ]
            ),
            agr_report,
        )
    # no agreements nor disagreements found for the pre-specified rules
    return 1.0, {}

def checkAgreementScores(lang_rule_all, token, sent, featuresInDatapoint, agreement_aggr):
    task = 'agreement'
    if task in lang_rule_all:
        rulesPerAgreement = lang_rule_all[task]

        agreement_rules_per_sent = {}
        for model, rules in rulesPerAgreement.items(): #Gender:[], Person:[], Number:[]
            obsAgreement = utils.checkModelApplicable(task, model, token, sent)
            if obsAgreement != -1: #-1 denotes that rule is not applicable to this datapoint e.g. for testing Gender agreement, gender is not present
                active_features, nonactive_features, labels = utils.extractFeaturesFromRules(rules)
                for one_rule_active, one_rule_nonactive, label in zip(active_features, nonactive_features, labels):
                    # one_rule_active contains the active features for this rule
                    label = 1 #For agreement we only retain rules for required-agreement, so label is always set to 1
                    if utils.isGrammarRuleApplicable(featuresInDatapoint, one_rule_active, one_rule_nonactive, prop=model):
                        agr_type = "%s-%s-%s" % (token.deprel, token.upos, sent[token.head].upos) #rel, dep, head
                        if agr_type not in agreement_aggr:
                            agreement_aggr[agr_type] = {}
                        if model not in agreement_aggr[agr_type]:
                            agreement_aggr[agr_type][model] = [0] * 3

                        if obsAgreement == label:
                            agreement_aggr[agr_type][model][1] +=1

                        else:
                            agreement_aggr[agr_type][model][0] +=1
                            agreement_rules_per_sent[model] = (one_rule_active, one_rule_nonactive, 'req-agree')
                        agreement_aggr[agr_type][model][2] +=1



        return agreement_rules_per_sent


def checkWordOrderScores(lang_rule_all, token, sent, featuresInDatapoint, wordorder_aggr):
    task = 'wordorder'
    if task in lang_rule_all:
        rulesPerWordOrder = lang_rule_all[task]

        wordorder_rules_per_sent = {}
        for model, rules in rulesPerWordOrder.items(): #subject-verb:[], object-verb:[], adjective-noun:[], noun-adposition:[], numeral-noun:[]
            obsWordOrder = utils.checkModelApplicable(task, model, token, sent)
            if obsWordOrder != -1: #-1 denotes that rule is not applicable to this datapoint e.g. for testing subject-verb agreement, subj is not present
                active_features, nonactive_features, labels = utils.extractFeaturesFromRules(rules)
                for one_rule_active, one_rule_nonactive, label in zip(active_features, nonactive_features, labels):
                    # one_rule_active contains the active features for this rule
                    if utils.isGrammarRuleApplicable(featuresInDatapoint, one_rule_active, one_rule_nonactive):
                        if model not in wordorder_aggr:
                            wordorder_aggr[model] = [0] * 3

                        if obsWordOrder == label:
                            wordorder_aggr[model][1] +=1

                        else:
                            wordorder_aggr[model][0] +=1
                            wordorder_rules_per_sent[model] = (one_rule_active, one_rule_nonactive, label)
                        wordorder_aggr[model][2] +=1



        return wordorder_rules_per_sent


def checkAssignmentScores(lang_rule_all, token, sent, featuresInDatapoint, assignment_aggr, argstruct_aggr):
    task = 'casemarking'
    if task in lang_rule_all:
        rulesPerAssignment = lang_rule_all[task]

        assignment_rules_per_sent = {}
        for model, rules in rulesPerAssignment.items(): #NOUN:[], PROPN:[], PRON:[]
            obsCase = utils.checkModelApplicable(task, model, token, sent)
            if obsCase != -1: #-1 denotes that rule is not applicable to this datapoint e.g. for testing subject-verb agreement, subj is not present
                if token.head =='0' and token.head:
                    agr_type = None
                else:
                    agr_type = "%s-%s-%s" % (token.deprel, token.upos, sent[token.head].upos)  # rel, dep, head

                active_features, nonactive_features, labels = utils.extractFeaturesFromRules(rules)
                for one_rule_active, one_rule_nonactive, label in zip(active_features, nonactive_features, labels):
                    # one_rule_active contains the active features for this rule
                    if utils.isGrammarRuleApplicable(featuresInDatapoint, one_rule_active, one_rule_nonactive):
                        if agr_type and agr_type not in argstruct_aggr:
                            argstruct_aggr[agr_type] = {}
                            argstruct_aggr[agr_type]['depd'] =  {'feat_value': label, "counts": [0, 0, 0]}

                        if model not in assignment_aggr:
                            assignment_aggr[model] = [0] * 3

                        if obsCase == label:
                            assignment_aggr[model][1] +=1
                            if agr_type:
                                argstruct_aggr[agr_type]['depd']['counts'][1] += 1

                        else:
                            assignment_aggr[model][0] +=1
                            assignment_rules_per_sent[model] = (one_rule_active, one_rule_nonactive, label)
                            if agr_type:
                                argstruct_aggr[agr_type]['depd']['counts'][0] += 1

                        assignment_aggr[model][2] +=1

                        if agr_type:
                            argstruct_aggr[agr_type]['depd']['counts'][2] += 1



        return assignment_rules_per_sent


def get_sent_score_all(data, lang_rule_all):
    """
    computes the grammar error metric at sentence level
    """

    scores = []
    for sent in data:
        agreement_aggr = {}
        wordorder_aggr = {}
        assignment_aggr = {}
        argstruct_aggr = {}

        # Add the head-dependents
        dep_data_token = defaultdict(list)
        sent_tokens = []
        id2index = sent._ids_to_indexes
        for token_num, token in enumerate(sent):
            dep_data_token[token.head].append(token.id)
            sent_tokens.append(token.form)


        for token in sent:
            featuresInDatapoint = utils.extractFeatures(token_num, token, sent, dep_data_token, use_lexical=True)

            # Checking agreement for Gender, Person, Number
            checkAgreementScores(lang_rule_all, token, sent, featuresInDatapoint,
                                                                agreement_aggr)

            # Checking word order for subject-verb, object-verb, adj-noun, noun-adp, numeral-noun
            checkWordOrderScores(lang_rule_all, token, sent, featuresInDatapoint,
                                                                wordorder_aggr)

            # Checking casemarking for nouns, propernouns, pronouns,
            checkAssignmentScores(lang_rule_all, token, sent, featuresInDatapoint,
                                                                  assignment_aggr, argstruct_aggr)

        agr_score, agr_report = compute_score(agreement_aggr, task='agreement', argstruct_aggr=None)
        wo_score, wo_report = compute_score(wordorder_aggr, task='wordorder', argstruct_aggr=None)
        am_score, am_report = compute_score(assignment_aggr, task='casemarking', argstruct_aggr=argstruct_aggr)

        score, report = compute_joint_score(agreement_aggr, wordorder_aggr, assignment_aggr, argstruct_aggr)
        scores.append(
            {
                "agr": agr_score,
                "agr_report": agr_report,
                "wo": wo_score,
                "wo_report": wo_report,
                "assignment": am_score,
                "assignment_report": am_report,
                "joint": score,
                "joint_report": report,
            }
        )

    return scores


def get_doc_score_all(data, lang_rule_all):
    """
    computes grammar error metric at document level
    """

    agreement_aggr = {}
    argstruct_aggr = {}
    wordorder_aggr = {}
    assignment_aggr = {}
    # for depd_type in lang_argstruct:
    #     argstruct_aggr[depd_type] = {}
    #     for feat in lang_argstruct[depd_type]:
    #         depd_feat_value, head_feat_value = lang_argstruct[depd_type][feat]
    #         argstruct_aggr[depd_type][feat] = {
    #             "depd": {"feat_value": depd_feat_value, "counts": [0, 0, 0],},
    #             "head": {"feat_value": head_feat_value, "counts": [0, 0, 0],},
    #         }
    with open(f'{args.output}/{args.lg}.examples', 'w') as fout:
        for sent in data:

            # Add the head-dependents
            dep_data_token = defaultdict(list)
            sent_tokens = []
            id2index = sent._ids_to_indexes
            for token_num, token in enumerate(sent):
                dep_data_token[token.head].append(token.id)
                sent_tokens.append(token.form)

            for token_num, token in enumerate(sent):

                featuresInDatapoint = utils.extractFeatures(token_num, token, sent, dep_data_token, use_lexical=True)

                #Checking agreement for Gender, Person, Number
                agreement_rules_not_followed = checkAgreementScores(lang_rule_all, token, sent, featuresInDatapoint, agreement_aggr)
                utils.printExamples(agreement_rules_not_followed, sent_tokens, token, token_num, sent, id2index, fout, task='agreement')

                #Checking word order for subject-verb, object-verb, adj-noun, noun-adp, numeral-noun
                wordorder_rules_not_followed = checkWordOrderScores(lang_rule_all, token, sent, featuresInDatapoint, wordorder_aggr)
                utils.printExamples(wordorder_rules_not_followed, sent_tokens, token, token_num, sent, id2index, fout,
                                    task='wordorder')

                # Checking casemarking for nouns, propernouns, pronouns,
                assignment_rules_not_followed = checkAssignmentScores(lang_rule_all, token, sent, featuresInDatapoint,
                                                                    assignment_aggr, argstruct_aggr)
                utils.printExamples(assignment_rules_not_followed, sent_tokens, token, token_num, sent, id2index, fout,
                                    task='casemarking')

    score, report = compute_joint_score(agreement_aggr, wordorder_aggr, assignment_aggr, argstruct_aggr)
    agr_score, agr_report = compute_score(agreement_aggr, task='agreement', argstruct_aggr=None)
    wo_score, wo_report = compute_score(wordorder_aggr, task='wordorder', argstruct_aggr=None)
    argstruct_score, argstruct_report = compute_score(assignment_aggr, task="casemarking", argstruct_aggr=argstruct_aggr)

    score_dict = {
        "agr_score": agr_score,
        "agr_report": agr_report,
        "wo_score": wo_score,
        "wo_report": wo_report,
        "argstruct_score": argstruct_score,
        "argstruct_report": argstruct_report,
        "joint_score": score,
        "joint_report": report,
    }
    return score_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="compute morphological well-formedness"
    )
    parser.add_argument("-input", type=str, help="input CoNLL-U file", default="../data/conllu/tr.conllu")
    parser.add_argument(
        "-lg", type=str, help="input language ISO 639-1 code", default="tr"
    )
    parser.add_argument(
        "-all", type=str, help="all rules file containing rules for agreement, word order, case marking", default="../rules/all_rules.txt"
    )
    parser.add_argument("--report", action="store_true", default=True)
    parser.add_argument("--output", type=str, help="Output the examples where the grammar rules don't follow", default="../data/outputs/")
    args = parser.parse_args()

    lang_rule_all = utils.load_rule_file(args.all)
    if args.lg not in lang_rule_all:
        print(f"No rules found for {args.lg}, exiting!")
        exit(-1)

    lang_rule_all = lang_rule_all[args.lg]
    sentences = pyconll.load_from_file(args.input)
    doc_score = get_doc_score_all(sentences, lang_rule_all)

    print(f"score: {doc_score['joint_score']:.4f}")
    if args.report:
        doc_report = doc_score["joint_report"]
        for rule, score in doc_report.items():
            print(f"{rule}\t{score:.4f}")
