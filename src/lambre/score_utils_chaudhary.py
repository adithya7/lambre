import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import lambre.rule_utils as utils


def compute_joint_score(
    agreement_aggr, wordorder_aggr, assignment_aggr, argstruct_aggr
):

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
                report["agr = %s:%s" % (agr_type, dim)] = agr_score

    for dim in wordorder_aggr:
        mismatch, match, total = wordorder_aggr[dim]
        if mismatch + match > 0:
            agr_score, agr_weight = float(match) / (match + mismatch), 1
            weights.append(agr_weight)
            scores.append(agr_score)
            dim_score_match[dim] += match
            dim_score_total[dim] += total
            report["wo = %s" % (dim)] = agr_score

    for dim in assignment_aggr:
        mismatch, match, total = assignment_aggr[dim]
        if mismatch + match > 0:
            agr_score, agr_weight = float(match) / (match + mismatch), 1
            weights.append(agr_weight)
            scores.append(agr_score)
            dim_score_match[dim] += match
            dim_score_total[dim] += total
            report["assignment = %s" % (dim)] = agr_score

    for depd_type in argstruct_aggr:  # we only do for Case
        for token_type in argstruct_aggr[depd_type]:
            mismatch, match, total = argstruct_aggr[depd_type][token_type]["counts"]
            if mismatch + match > 0 and total > 0:
                score, weight = float(match) / (match + mismatch), 1
                weights.append(weight)
                scores.append(score)
                report["args = %s:%s:%s" % (depd_type, token_type, "Case")] = score

    for dim, match in dim_score_match.items():
        total = dim_score_total[dim]
        percentage_match = float(match) / total
        report[f"model: {dim}"] = percentage_match

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

    if task == "agreement":
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
    elif task == "wordorder":
        for dim in aggr:
            mismatch, match, total = aggr[dim]
            if mismatch + match > 0:
                agr_score, agr_weight = float(match) / (match + mismatch), 1
                weights.append(agr_weight)
                scores.append(agr_score)
                dim_score_match[dim] += match
                dim_score_total[dim] += total
                agr_report["wordorder = %s" % (dim)] = agr_score * 100.0
    elif task == "casemarking":
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
                mismatch, match, total = argstruct_aggr[depd_type][token_type]["counts"]
                if mismatch + match > 0 and total > 0:
                    score, weight = float(match) / (match + mismatch), 1
                    # weights.append(weight)
                    # scores.append(score)
                    agr_report["args=%s:%s:%s" % (depd_type, token_type, "Case")] = (
                        score * 100.0
                    )

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


def checkAgreementScores(
    lang_rule_all, token, sent, featuresInDatapoint, agreement_aggr, sent_agreement_aggr
):
    task = "agreement"
    if task in lang_rule_all:
        rulesPerAgreement = lang_rule_all[task]

        agreement_rules_per_sent, error = {}, False
        for (
            model,
            rules,
        ) in rulesPerAgreement.items():  # gender-NOUN:[], Person:[], Number:[]
            # model_feature = model.split("-")[0].title()
            # if model_feature not in agreement_rules_per_sent:
            agreement_rules_per_sent[model] = []
            obsAgreement = utils.checkModelApplicable(task, model, token, sent)
            if (
                obsAgreement != -1
            ):  # -1 denotes that rule is not applicable to this datapoint e.g. for testing Gender agreement, gender is not present
                (
                    active_features,
                    nonactive_features,
                    labels,
                ) = utils.extractFeaturesFromRules(rules)
                for one_rule_active, one_rule_nonactive, label in zip(
                    active_features, nonactive_features, labels
                ):
                    # one_rule_active contains the active features for this rule
                    label = 1  # For agreement we only retain rules for required-agreement, so label is always set to 1
                    if utils.isGrammarRuleApplicable(
                        featuresInDatapoint,
                        one_rule_active,
                        one_rule_nonactive,
                        prop=model,
                    ):
                        agr_type = "%s-%s-%s" % (
                            token.deprel,
                            token.upos,
                            sent[token.head].upos,
                        )  # rel, dep, head
                        if agr_type not in agreement_aggr:
                            agreement_aggr[agr_type] = {}
                        if agr_type not in sent_agreement_aggr:
                            sent_agreement_aggr[agr_type] = {}

                        if model not in agreement_aggr[agr_type]:
                            agreement_aggr[agr_type][model] = [0] * 3
                        if model not in sent_agreement_aggr[agr_type]:
                            sent_agreement_aggr[agr_type][model] = [0] * 3

                        if obsAgreement == label:
                            agreement_aggr[agr_type][model][1] += 1
                            sent_agreement_aggr[agr_type][model][1] += 1

                        else:
                            agreement_aggr[agr_type][model][0] += 1
                            agreement_rules_per_sent[model].append(
                                (
                                    one_rule_active,
                                    one_rule_nonactive,
                                    "req-agree",
                                )
                            )
                            sent_agreement_aggr[agr_type][model][0] += 1
                            error = True
                        agreement_aggr[agr_type][model][2] += 1
                        sent_agreement_aggr[agr_type][model][2] += 1

        return agreement_rules_per_sent, error

    return None, False


def checkWordOrderScores(
    lang_rule_all, token, sent, featuresInDatapoint, wordorder_aggr, sent_wordorder_aggr
):
    task = "wordorder"
    if task in lang_rule_all:
        error = False
        rulesPerWordOrder = lang_rule_all[task]

        wordorder_rules_per_sent = {}
        for (
            model,
            rules,
        ) in (
            rulesPerWordOrder.items()
        ):  # subject-verb:[], object-verb:[], adjective-noun:[], noun-adposition:[], numeral-noun:[]
            wordorder_rules_per_sent[model] = []
            obsWordOrder = utils.checkModelApplicable(task, model, token, sent)
            if (
                obsWordOrder != -1
            ):  # -1 denotes that rule is not applicable to this datapoint e.g. for testing subject-verb agreement, subj is not present
                (
                    active_features,
                    nonactive_features,
                    labels,
                ) = utils.extractFeaturesFromRules(rules)
                for one_rule_active, one_rule_nonactive, label in zip(
                    active_features, nonactive_features, labels
                ):
                    # one_rule_active contains the active features for this rule
                    if utils.isGrammarRuleApplicable(
                        featuresInDatapoint, one_rule_active, one_rule_nonactive
                    ):
                        if model not in wordorder_aggr:
                            wordorder_aggr[model] = [0] * 3
                        if model not in sent_wordorder_aggr:
                            sent_wordorder_aggr[model] = [0] * 3

                        if obsWordOrder == label:
                            wordorder_aggr[model][1] += 1
                            sent_wordorder_aggr[model][1] += 1

                        else:
                            wordorder_aggr[model][0] += 1
                            wordorder_rules_per_sent[model].append(
                                (one_rule_active, one_rule_nonactive, label)
                            )
                            sent_wordorder_aggr[model][0] += 1
                            error = True
                        wordorder_aggr[model][2] += 1
                        sent_wordorder_aggr[model][2] += 1

        return wordorder_rules_per_sent, error

    return None, False


def checkAssignmentScores(
    lang_rule_all,
    token,
    sent,
    featuresInDatapoint,
    assignment_aggr,
    argstruct_aggr,
    sent_assignment_aggr,
):
    task = "casemarking"
    if task in lang_rule_all:
        error = False
        rulesPerAssignment = lang_rule_all[task]

        assignment_rules_per_sent = {}
        for model, rules in rulesPerAssignment.items():  # NOUN:[], PROPN:[], PRON:[]
            obsCase = utils.checkModelApplicable(task, model, token, sent)
            if (
                obsCase != -1
            ):  # -1 denotes that rule is not applicable to this datapoint e.g. for testing subject-verb agreement, subj is not present
                assignment_rules_per_sent[model] = []
                if token.head == "0" and token.head:
                    agr_type = None
                else:
                    agr_type = "%s-%s-%s" % (
                        token.deprel,
                        token.upos,
                        sent[token.head].upos,
                    )  # rel, dep, head

                (
                    active_features,
                    nonactive_features,
                    labels,
                ) = utils.extractFeaturesFromRules(rules)
                for one_rule_active, one_rule_nonactive, label in zip(
                    active_features, nonactive_features, labels
                ):
                    # one_rule_active contains the active features for this rule
                    if utils.isGrammarRuleApplicable(
                        featuresInDatapoint, one_rule_active, one_rule_nonactive
                    ):
                        if agr_type and agr_type not in argstruct_aggr:
                            argstruct_aggr[agr_type] = {}
                            argstruct_aggr[agr_type]["depd"] = {
                                "feat_value": label,
                                "counts": [0, 0, 0],
                            }

                        if model not in assignment_aggr:
                            assignment_aggr[model] = [0] * 3
                        if model not in sent_assignment_aggr:
                            sent_assignment_aggr[model] = [0] * 3

                        if obsCase == label:
                            assignment_aggr[model][1] += 1
                            sent_assignment_aggr[model][1] += 1
                            if agr_type:
                                argstruct_aggr[agr_type]["depd"]["counts"][1] += 1

                        else:
                            assignment_aggr[model][0] += 1
                            sent_assignment_aggr[model][0] += 1
                            assignment_rules_per_sent[model].append(
                                (one_rule_active, one_rule_nonactive, label)
                            )
                            if agr_type:
                                argstruct_aggr[agr_type]["depd"]["counts"][0] += 1
                            error = True

                        assignment_aggr[model][2] += 1
                        sent_assignment_aggr[model][2] += 1

                        if agr_type:
                            argstruct_aggr[agr_type]["depd"]["counts"][2] += 1

        return assignment_rules_per_sent, error

    return None, False


def get_sent_score(data, lang_rule_all, verbose: bool = False):
    """
    computes the grammar error metric at sentence level
    """

    logging.info(f"computing sentence-level lambre score")

    scores = []
    sent_error_examples = []

    for sent in tqdm(data, disable=not verbose):
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
            featuresInDatapoint = utils.extractFeatures(
                token_num, token, sent, dep_data_token, use_lexical=True
            )

            # Checking agreement for Gender, Person, Number
            agreement_rules_not_followed, isAgreeError = checkAgreementScores(
                lang_rule_all, token, sent, featuresInDatapoint, agreement_aggr, {}
            )

            # Checking word order for subject-verb, object-verb, adj-noun, noun-adp, numeral-noun
            wordorder_rules_not_followed, isWordOrderError = checkWordOrderScores(
                lang_rule_all, token, sent, featuresInDatapoint, wordorder_aggr, {}
            )

            # Checking casemarking for nouns, propernouns, pronouns,
            assignment_rules_not_followed, isAssignmentError = checkAssignmentScores(
                lang_rule_all,
                token,
                sent,
                featuresInDatapoint,
                assignment_aggr,
                argstruct_aggr,
                {},
            )

            if isAgreeError or isWordOrderError or isAssignmentError:
                sent_error_examples += [
                    (
                        sent,
                        token,
                        isAgreeError,
                        isWordOrderError,
                        isAssignmentError,
                        agreement_rules_not_followed,
                        wordorder_rules_not_followed,
                        assignment_rules_not_followed,
                    )
                ]

        agr_score, agr_report = compute_score(
            agreement_aggr, task="agreement", argstruct_aggr=None
        )
        wo_score, wo_report = compute_score(
            wordorder_aggr, task="wordorder", argstruct_aggr=None
        )
        am_score, am_report = compute_score(
            assignment_aggr, task="casemarking", argstruct_aggr=argstruct_aggr
        )

        score, report = compute_joint_score(
            agreement_aggr, wordorder_aggr, assignment_aggr, argstruct_aggr
        )
        scores.append(
            {
                "agr_score": agr_score,
                "agr_report": agr_report,
                "wo_score": wo_score,
                "wo_report": wo_report,
                "assignment_score": am_score,
                "assignment_report": am_report,
                "joint_score": score,
                "joint_report": report,
                "sent": " ".join(sent_tokens),
            }
        )

    return scores, sent_error_examples


def get_doc_score(data, lang_rule_all, verbose: bool = False):
    """
    computes grammar error metric at document level
    """

    logging.info(f"computing document-level lambre score")

    agreement_aggr = {}
    argstruct_aggr = {}
    wordorder_aggr = {}
    assignment_aggr = {}
    sent_error_examples = []
    for sent in tqdm(data, disable=not verbose):

        # Add the head-dependents
        dep_data_token = defaultdict(list)
        sent_tokens = []
        id2index = sent._ids_to_indexes
        for token_num, token in enumerate(sent):
            dep_data_token[token.head].append(token.id)
            sent_tokens.append(token.form)

        sent_agreement_aggr = {}
        sent_argstruct_aggr = {}
        sent_wordorder_aggr = {}
        sent_assignment_aggr = {}
        for token_num, token in enumerate(sent):

            featuresInDatapoint = utils.extractFeatures(
                token_num, token, sent, dep_data_token, use_lexical=True
            )

            # Checking agreement for Gender, Person, Number
            agreement_rules_not_followed, isAgreeError = checkAgreementScores(
                lang_rule_all,
                token,
                sent,
                featuresInDatapoint,
                agreement_aggr,
                sent_agreement_aggr,
            )

            # Checking word order for subject-verb, object-verb, adj-noun, noun-adp, numeral-noun
            wordorder_rules_not_followed, isWordOrderError = checkWordOrderScores(
                lang_rule_all,
                token,
                sent,
                featuresInDatapoint,
                wordorder_aggr,
                sent_wordorder_aggr,
            )
            # utils.printExamples(
            #     wordorder_rules_not_followed,
            #     sent_tokens,
            #     token,
            #     token_num,
            #     sent,
            #     id2index,
            #     sent_error_examples,
            #     task="wordorder",
            # )

            # Checking casemarking for nouns, propernouns, pronouns,
            assignment_rules_not_followed, isAssignmentError = checkAssignmentScores(
                lang_rule_all,
                token,
                sent,
                featuresInDatapoint,
                assignment_aggr,
                argstruct_aggr,
                sent_assignment_aggr,
            )
            # utils.printExamples(
            #     assignment_rules_not_followed,
            #     sent_tokens,
            #     token,
            #     token_num,
            #     sent,
            #     id2index,
            #     sent_error_examples,
            #     task="casemarking",
            # )

            if isAgreeError or isWordOrderError or isAssignmentError:
                sent_error_examples += [
                    (
                        sent,
                        token,
                        isAgreeError,
                        isWordOrderError,
                        isAssignmentError,
                        agreement_rules_not_followed,
                        wordorder_rules_not_followed,
                        assignment_rules_not_followed,
                    )
                ]

        sent_score, sent_report = compute_joint_score(
            sent_agreement_aggr,
            sent_wordorder_aggr,
            sent_assignment_aggr,
            sent_argstruct_aggr,
        )
        # fout.write(f'sent score: {sent_score} \t sent: {" ".join(sent_tokens)}\n')
        # sent_score_examples.append((sent_score, sent_error_examples, " ".join(sent_tokens)))
        # sorted error sents
        # sent_score_examples.sort()
        # for (sent_score, sent_error_examples, sent) in sent_score_examples[:500]:
        #     fout.write(f"score: {sent_score} \t sent: {sent}\n")
        #     fout.write(f'{"".join(sent_error_examples)}\n\n')
    score, report = compute_joint_score(
        agreement_aggr, wordorder_aggr, assignment_aggr, argstruct_aggr
    )
    agr_score, agr_report = compute_score(
        agreement_aggr, task="agreement", argstruct_aggr=None
    )
    wo_score, wo_report = compute_score(
        wordorder_aggr, task="wordorder", argstruct_aggr=None
    )
    argstruct_score, argstruct_report = compute_score(
        assignment_aggr, task="casemarking", argstruct_aggr=argstruct_aggr
    )

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
    return score_dict, sent_error_examples
