from collections import defaultdict
from copy import deepcopy
from pathlib import Path


def load_pratapa_etal_2021_rules(file_path: Path):
    """
    load rules from Pratapa et al., 2021 for a specific language
    """

    agr_rules = defaultdict(list)
    argstruct_rules = {}
    with open(file_path, "r") as rf:
        for line in rf:
            splits = line.strip().split("\t")
            rule_type = splits[2]
            if rule_type == "agreement":
                depd_rel, feat = splits[3], splits[4]
                agr_rules[depd_rel] += [feat]
            elif rule_type == "argstruct":
                depd_rel, feat, depd_feat_val, head_feat_val = splits[3:7]
                if depd_rel not in argstruct_rules:
                    argstruct_rules[depd_rel] = {}
                argstruct_rules[depd_rel][feat] = (
                    depd_feat_val.replace(";", ","),
                    head_feat_val.replace(";", ","),
                )
    return agr_rules, argstruct_rules


def load_chaudhury_etal_2021_rules(file_path: Path):
    """
    load rules from Chaudhury et al., 2021 for a specific language
    """

    rules = {}
    with open(file_path, "r") as rf:
        # tr      chaudhary-etal-2021     wordorder       subject-verb  before      depheadpos_NOUN ### deppos_VERB depheadpos_PUNCT ### headpos_VERB       71
        # tr      chaudhary-etal-2021     agreement       gender-AUX req-agree       depheadpos_NOUN ### deppos_VERB depheadpos_PUNCT ### headpos_VERB       71
        for line in rf:
            line_txt = line.strip().split("\t")
            lang, _, task, model, label, active, non_active, total_samples = line_txt
            if task not in rules:
                rules[task] = defaultdict(list)

            rules[task][model].append((label, active, non_active, total_samples))

    return rules


def isPropertyPresent(prop, token, isHead=False):
    if token is None:
        return False

    if not isHead:  # Checking if the head of the the dependent is no root
        if token.head == "0":
            return False

    if prop in token.feats:
        return True

    return False


def extractFeaturesFromRules(rules):
    active_features, nonactive_features, labels = [], [], []

    for (label, active, non_active, _) in rules:
        active = active.split(" ### ")
        non_active = non_active.split(" ### ")

        one_active, one_nonactive = [], []
        for f in active:
            if f != "NA":
                one_active.append(f)

        for f in non_active:
            if f != "NA":
                one_nonactive.append(f)

        active_features.append(one_active)
        nonactive_features.append(one_nonactive)
        labels.append(label)

    return active_features, nonactive_features, labels


def extractFeatures(token_num, token, sentence, dep_data_token, use_lexical=False):
    features = []
    pos = token.upos
    feats = token.feats
    lemma = token.lemma

    feature = f"deppos_{pos}"
    features.append(feature)

    if token.deprel:
        relation = token.deprel.lower()
        feature = f"deprel_{relation}"
        features.append(feature)

    for feat in feats:
        feature = f"depfeat_{feat}_"
        value = getFeatureValue(feat, feats)
        feature += f"{value}"
        features.append(feature)

    if use_lexical:
        lemma = isValidLemma(lemma, pos)
        if lemma:
            feature = f"lemma_{lemma}"
            features.append(feature)

        # Add tokens in the neighborhood of 3
        neighboring_tokens_left = max(0, token_num - 3)
        neighboring_tokens_right = min(token_num + 3, len(sentence))
        for neighor in range(neighboring_tokens_left, neighboring_tokens_right):
            if neighor == token_num and neighor >= len(sentence):
                continue
            neighor_token = sentence[neighor]
            if neighor_token:
                lemma = isValidLemma(neighor_token.lemma, neighor_token.upos)
                if lemma:
                    feature = f"neighborhood_{lemma}"
                    features.append(feature)

    if token.head != "0" and token.head is not None:
        head_pos = sentence[token.head].upos
        headrelation = sentence[token.head].deprel
        head_feats = sentence[token.head].feats
        headhead = sentence[token.head].head
        head_lemma = sentence[token.head].lemma

        feature = f"headpos_{head_pos}"
        features.append(feature)

        if headrelation and headrelation != "root" and headrelation != "punct":
            feature = f"headrelrel_{headrelation.lower()}"
            features.append(feature)

            feature = f"headrelrel_{head_pos}_{relation}_{headrelation.lower()}"
            features.append(feature)

            feature = f"headrelrel_{head_pos}_{headrelation.lower()}"
            features.append(feature)

        for feat in head_feats:  # Adding features for dependent token (maybe more commonly occurring)
            feature = f"headfeat_{head_pos}_{feat}_"
            value = getFeatureValue(feat, head_feats)
            feature += f"{value}"
            features.append(feature)

            feature = f"headfeat_{feat}_{value}"
            features.append(feature)

            feature = f"headfeat_{head_pos}_{relation}_{feat}_"
            value = getFeatureValue(feat, head_feats)
            feature += f"{value}"
            features.append(feature)

            feature = f"headfeat_{head_pos}_{feat}_{value}"
            features.append(feature)

            feature = f"headfeatrel_{relation}_{feat}_{value}"
            features.append(feature)

        if headhead and headhead != "0":

            headhead_feats = sentence[headhead].feats
            for prop in ["Gender", "Person", "Number"]:
                headlabel = getFeatureValue(prop, head_feats)
                if headlabel and prop in headhead_feats:
                    headhead_value = getFeatureValue(prop, headhead_feats)
                    if headlabel == headhead_value:
                        feature = f"headmatch_True_{prop}"
                        features.append(feature)

            headheadheadlemma = isValidLemma(sentence[headhead].lemma, sentence[headhead].upos)
            if use_lexical and headheadheadlemma:
                feature = f"headheadlemma_{headheadheadlemma}"
                features.append(feature)

        if use_lexical:
            head_lemma = isValidLemma(head_lemma, head_pos)
            if head_lemma:
                feature = f"headlemma_{head_lemma}"
                features.append(feature)

        if "Case" in head_feats and "Case" in feats:
            label = getFeatureValue("Case", feats)
            headlabel = getFeatureValue("Case", head_feats)

            if label == headlabel:  # If agreement between the head-dep
                feature = f"agreepos_{head_pos}"
                features.append(feature)

                feature = f"agreerel_{relation}"
                features.append(feature)

                if headrelation and headrelation != "root" and headrelation != "punct":
                    feature = f"agree_{relation}_{head_pos}_{headrelation.lower()}"
                    features.append(feature)

                    feature = f"agree_{headrelation.lower()}"
                    features.append(feature)

    # get other dep tokens of the head
    dep = dep_data_token.get(token.head, [])
    for d in dep:
        if d == token.id:
            continue
        depdeprelation = sentence[d].deprel
        if depdeprelation and depdeprelation != "punct":
            feature = f"depheadrel_{depdeprelation}"
            features.append(feature)

        depdeppos = sentence[d].upos
        feature = f"depheadpos_{depdeppos}"
        features.append(feature)

        depdeplemma = isValidLemma(sentence[d].lemma, sentence[d].upos)
        if use_lexical and depdeplemma:
            feature = f"depheadlemma_{depdeplemma}"
            features.append(feature)

    # adding the children of the dep token
    for dep in dep_data_token[token.id]:
        deptoken = sentence[dep]
        feature = f"depdeppos_{deptoken.upos}"
        features.append(feature)

        deprel = deptoken.deprel
        if deprel and deprel != "root" and deprel != "punct":
            feature = f"depdeprel_{deprel}"
            features.append(feature)

        deplemma = isValidLemma(deptoken.lemma, deptoken.upos)
        if use_lexical and deplemma:
            feature = f"depdeplemma_{deplemma}"
            features.append(feature)

    return features


def getFeatureValue(feat, feats):
    if feat not in feats:
        return None
    values = list(feats[feat])
    values.sort()
    value = "/".join(values)
    return value


def isValidLemma(lemma, upos):
    if upos in ["PUNCT", "NUM", "PROPN" "X", "SYM"]:
        return None
    if lemma:
        lemma = lemma.lower()
        lemma = lemma.replace('"', "").replace("'", "")
        if lemma == "" or lemma == " ":
            return None
        else:
            return lemma
    return None


def checkModelApplicable(task, model, token, sent):
    if task == "agreement":
        # If the model (e.g. gender,person, number) in the head and dep, only then check for agreement match
        # model == gender-NOUN
        model_feature = model.split("-")[0].lower().title()
        pos = model.split("-")[1]
        if (
            isPropertyPresent(model_feature, token)
            and isPropertyPresent(model_feature, sent[token.head], isHead=True)
            and token.upos == pos
        ):
            dep_value = getFeatureValue(model, token.feats)
            head_value = getFeatureValue(model, sent[token.head].feats)

            if dep_value == head_value:  # observed agreement in the example
                return 1
            else:
                return 0
        else:
            return -1

    elif task == "wordorder":
        # the model is subject-verb for example
        pos = token.upos
        if token.head == "0" or not token.head:
            return -1
        head_pos = sent[token.head].upos
        relation = token.deprel

        wals_features = {
            "subject-verb": ["subj_VERB"],
            "object-verb": ["obj_VERB"],
            "noun-adposition": ["NOUN_ADP", "PRON_ADP", "PROPN_ADP"],  # When adp is the syntactic head
            "adjective-noun": ["ADJ_mod_NOUN", "ADJ_mod_PROPN", "ADJ_mod_PRON"],
            "numeral-noun": ["NUM_mod_NOUN", "NUM_mod_PROPN", "NUM_mod_PRON"],
        }
        defined_features = wals_features[model]
        isValid = False
        # Check if the model is applicable for this datapoint, e.g. for subject-verb check if the dep is a subj and its head is a verb
        for feature_type in defined_features:
            info = feature_type.split("_")
            if len(info) == 3:  # dep-pos-relation-head-pos
                if info[0] == pos and info[2] == head_pos and info[1] in relation:
                    isValid = True
                    break
            elif len(info) == 2:  # dep-relation, dep-head
                if info[0] in relation and info[1] == head_pos:
                    isValid = True
                    break
                if info[0] == pos and info[1] == head_pos:
                    isValid = True

        if isValid:
            # Get the observed label
            id2index = sent._ids_to_indexes
            token_position = id2index[token.id]
            head_position = id2index[token.head]
            if token_position < head_position:
                label = "before"
            else:
                label = "after"
            return label
        else:
            return -1

    elif task == "casemarking":
        # the model is casemarking for nouns for example
        pos = token.upos
        feats = token.feats
        if model != pos or "Case" not in feats:
            return -1
        label = getFeatureValue("Case", feats)
        return label


def isGrammarRuleApplicable(featuresInDatapoint, one_rule_active, one_rule_nonactive, prop=None):
    updated_featuresInDatapoint = []
    for f in featuresInDatapoint:
        if f.startswith("headmatch_True") and prop:
            fprop = f.split("_")[1]
            if prop == fprop:
                f = "headmatch_True"
            else:
                f = None

        if f:
            updated_featuresInDatapoint.append(f)

    valid = True
    # check if the active features defined in the rule are active for this datapoint
    for f in one_rule_active:
        if f and f not in updated_featuresInDatapoint:
            valid = False
            break

    if not valid:
        return False

    # check if any of the nonactive features in the rule are really not present for this datapoint
    for f in one_rule_nonactive:
        if f and f in updated_featuresInDatapoint:
            valid = False
            break

    return valid


def printExamples(
    rules_not_followed, sent_tokens, token, token_num, sent, id2index, sent_error_examples, task
):
    if len(rules_not_followed) > 0:  # there are rules which were not followed for this datapoint
        if task == "agreement":
            for model, (one_active, one_nonactive, label) in rules_not_followed.items():
                sent_example_tokens = deepcopy(sent_tokens)
                token_feature_value = getFeatureValue(model, token.feats)
                sent_example_tokens[token_num] = (
                    "***" + sent_example_tokens[token_num] + f"({model}={token_feature_value})***"
                )

                token_head_num = id2index[token.head]
                headtoken_feature_value = getFeatureValue(model, sent[token.head].feats)
                sent_example_tokens[token_head_num] = (
                    "***" + sent_example_tokens[token_head_num] + f"({model}={headtoken_feature_value})***"
                )

                sent_error_examples.append(f'Example: {" ".join(sent_example_tokens)}\n')
                sent_error_examples.append(
                    f"{model} agreement not followed by tokens marked *** because following rule was not satisfied:\n"
                )
                # if len(one_active) > 0:
                #     sent_error_examples.append(f'Required Active features in the rule: {" ### ".join(one_active)}\n')
                # if len(one_nonactive) > 0:
                #     sent_error_examples.append(f'Required Non-active features in that rule: {" ### ".join(one_nonactive)}\n')
                sent_error_examples.append("\n")

        elif task == "wordorder":
            for model, (one_active, one_nonactive, label) in rules_not_followed.items():
                dep, head = model.split("-")[0], model.split("-")[1]
                if head == "noun":
                    head = "nominal"
                if label == "before":
                    not_label = "after"
                else:
                    not_label = "before"

                sent_example_tokens = deepcopy(sent_tokens)
                sent_example_tokens[token_num] = "***" + sent_example_tokens[token_num] + f"({dep})***"

                token_head_num = id2index[token.head]
                sent_example_tokens[token_head_num] = (
                    "***" + sent_example_tokens[token_head_num] + f"({head})***"
                )

                sent_error_examples.append(f'Example: {" ".join(sent_example_tokens)}\n')
                sent_error_examples.append(
                    f"{model} order not followed for tokens marked ***, predicted order is {label} but observed is {not_label}. because following rule was not satisfied:\n"
                )
                # if len(one_active) > 0:
                #     sent_error_examples.append(f'Required Active features in the rule: {" ### ".join(one_active)}\n')
                # if len(one_nonactive) > 0:
                #     sent_error_examples.append(f'Required Non-active features in that rule: {" ### ".join(one_nonactive)}\n')
                sent_error_examples.append("\n")

        elif task == "casemarking":
            for model, (one_active, one_nonactive, label) in rules_not_followed.items():
                sent_example_tokens = deepcopy(sent_tokens)
                value = getFeatureValue("Case", token.feats)
                sent_example_tokens[token_num] = (
                    "***" + sent_example_tokens[token_num] + f"({token.upos}'s Case={value})***"
                )

                sent_error_examples.append(f'Example: {" ".join(sent_example_tokens)}\n')
                sent_error_examples.append(
                    f"{model} case marking not followed for tokens marked ***, predicted case is [{label}] but observed is [{value}], because following rule was not satisfied:\n"
                )
                # if len(one_active) > 0:
                #     sent_error_examples.append(f'Required Active features in the rule: {" ### ".join(one_active)}\n')
                # if len(one_nonactive) > 0:
                #     sent_error_examples.append(f'Required Non-active features in that rule: {" ### ".join(one_nonactive)}\n')
                sent_error_examples.append("\n")


def transformRulesIntoReadable(feature, task, rel, relation_map):
    task = task.lower()
    if task == "wordorder":
        dependent = rel.split("-")[0]
        head = rel.split("-")[1]
        if rel in ["adjective-noun", "numeral-noun"]:
            head = "nominal"
        if rel in ["noun-adposition"]:
            dependent = "nominal"
    elif task == "agreement":
        dependent = "dependent"
        head = "head"
    elif task == "assignment":
        dependent = rel
        head = "head"
    else:
        dependent = "current word"
        head = "head"

    new_features = feature
    info = feature.split("_")

    def get_relation(info):
        lang_link = f"https://universaldependencies.org/"
        if info in relation_map or info.lower() in relation_map:
            (value, _) = relation_map.get(info.lower(), info.lower())

        elif info.split("@")[0] in relation_map or info.split("@")[0].lower() in relation_map:
            (value, _) = relation_map.get(info.split("@")[0].lower(), info.lower())

        else:
            value = info
        return value, lang_link

    if feature.startswith("spinehead"):
        return f"{head} is a word like= {info[1]}"

    elif feature.startswith("spine"):
        return f"{dependent} is a word like= {info[1]}"

    if len(info) == 2:  # f'dep_{relation}_'
        if feature.startswith("wiki"):
            new_features = f"{dependent}'s semantic class is= {info[1]}"
        elif feature == "headmatch_True":
            new_features = f"the head agrees with its head on= {rel}"
        elif feature.startswith("lemma"):
            new_features = f"{dependent} has lemma= {info[1]}"
        elif feature.startswith("headlemma"):
            new_features = f"{dependent} is governed by= {info[1]}"
        elif feature.startswith("depdeplemma"):
            new_features = f"{dependent} is governing= {info[1]}"
        elif feature.startswith("headheadlemma"):
            new_features = f"{head} is governed by= {info[1]}"
        elif feature.startswith("depheadlemma"):
            new_features = f"{head} is also governing= {info[1]}"
        elif feature.startswith("neighborhood"):
            new_features = f"{dependent} is nearby= {info[1]}"
        elif feature.startswith("left"):
            new_features = f"before the {dependent} is= {info[1]}"
        elif feature.startswith("right"):
            new_features = f"after the {dependent} is= {info[1]}"
        elif feature.startswith("lang"):
            new_features = f"lang is= {info[1]}"
        elif feature.startswith("srclem"):
            new_features = f"in English is= {info[1]}"
        elif feature.startswith("srcpos"):
            info_, _ = get_relation(info[1])

            new_features = f"in English is= {info[1]}"
        else:  # f'deppos_{pos}'

            info_, link = get_relation(info[1])

            if feature.startswith("deppos"):
                new_features = f"{dependent} is a= {info[1]}"

            elif feature.startswith("headpos"):
                new_features = f"{dependent} is governed by a= {info[1]}"

            elif feature.startswith("deprel"):
                info[1] = f'<a href="{link}" title=""> {info_} </a>'
                new_features = f"{dependent} is the= {info[1]}"

            elif feature.startswith("headrelrel"):
                info[1] = f'<a href="{link}" title=""> {info_} </a>'
                new_features = f"{dependent} is governed by= {info[1]}"

            elif feature.startswith("depdeppos"):
                new_features = f"{dependent} is governing= {info[1]}"

            elif feature.startswith("depdeprel"):
                info[1] = f'<a href="{link}" title=""> {info_} </a>'
                new_features = f"{dependent} is governing= {info[1]}"

            elif feature.startswith("depheadrel") or feature.startswith("depheadpos"):
                head_phrase = head
                if head_phrase == "head":
                    new_features = f"***-marked token is nearby= {info[1]}"
                else:
                    new_features = f"***-marked token is nearby= {info[1]}"

            elif feature.startswith("svo"):
                new_features = f"anaphora= True"

            elif feature.startswith("agreepos"):
                new_features = f"{dependent} agrees with its head= {info[1]}"

            elif feature.startswith("agreerel"):
                info[1] = f'<a href="{link}" title=""> {info_} </a>'
                new_features = f"{dependent} agrees with its head when {dependent} is a= {info[1]}"

            elif feature.startswith("agree"):
                new_features = f"{dependent} agrees with its head when the head is a= {info[1]}"

    if len(info) == 3:  # f'depposrel_{pos}_{relation}'
        info_, link = get_relation(info[-1])
        if feature.startswith("depposrel"):
            info_pos, pos_link = get_relation(info[1])
            info[1] = info_pos
            info[2] = info_
            new_features = f"{info[1]} is the dependent and is the= {info[2]}"

        elif feature.startswith("headrelrel"):  # f'headrelrel_{head_pos}_{headrelation.lower()}'
            new_features = f"{dependent} has head-{info[1]} is a= {info_}"

        elif feature.startswith("headfeat"):
            new_features = f"{dependent} is governed by a word with {info[1]}= {info[2]}"

        elif feature.startswith("depfeat"):
            new_features = f"{dependent} with {info[1]}= {info[2]}"

        elif feature.startswith("wiki"):
            new_features = f"the head of the {dependent} has semantic class= {info[-1]}"

        elif feature.startswith("head"):  # f'head_{head_pos}_{relation}'
            info_pos, pos_link = get_relation(info[1])
            info[-1] = info_
            new_features = f"{info[-1]} of head= {info[1]}"

        elif feature.startswith("agree"):  # f'agree_{head_pos}_{headrelation.lower()}'
            info[-1] = info_
            new_features = f"{dependent} agrees with its {info[1]}-head which is a= {info[-1]}"

    if len(info) == 4:
        if feature.startswith("wiki"):
            new_features = f"the head has semantic class= {info[-1]}"

        elif feature.startswith("depfeat"):
            info[1] = info_pos
            new_features = (
                f"{info[1]} is the dependent with {info[2]}= {info[3]}"  # f'depfeat_{pos}_{feat}_{value}'
            )

        elif feature.startswith("headfeatrel"):  # f'headfeatrel_{rel}_{feat}_{value}'
            info_rel, rel_link = get_relation(info[1])
            info[1] = info_rel
            info[2] = info
            new_features = f"{dependent} has head is a {info[1]} with {info[2]}= {info[3]}"

        elif feature.startswith("headfeat"):  # f'headfeat_{head_pos}_{feat}_'
            info_pos, pos_link = get_relation(info[1])
            info[1] = info_pos
            new_features = f"{dependent} is governed by {info[1]} with {info[2]}= {info[3]}"

        elif feature.startswith("headrelrel"):  # f'headrelrel_{head_pos}_{relation}_{headrelation.lower()}'
            info_rel, rel_link = get_relation(info[-1])
            info[-1] = info_rel

            info_pos, pos_link = get_relation(info[1])
            info[1] = info_pos

            info_rel, rel_link = get_relation(info[2])
            info[2] = info_rel
            new_features = (
                f"{dependent} is a {info[2]} of {info[1]} where the head-{info[1]} is a= {info[-1]}"
            )

        elif feature.startswith("head"):  # f'head_{pos}_{relation}_{head}'
            info_rel, rel_link = get_relation(info[-1])
            info[-1] = info_rel

            info_pos, pos_link = get_relation(info[1])
            info[1] = info_pos

            info_pos, pos_link = get_relation(info[2])
            info[2] = info_pos
            new_features = f"{dependent} is a {info[1]} is a {info[2]} of head= {info[-1]}"

        elif feature.startswith("agree"):  # f'agree_{relation}_{head_pos}_{headrelation.lower()}'
            info_rel, rel_link = get_relation(info[-1])
            info[-1] = info_rel

            info_pos, pos_link = get_relation(info[1])
            info[1] = info_pos

            info_rel, rel_link = get_relation(info[2])
            info[2] = info_rel

            new_features = f"{dependent} is a {info[1]} and agrees with {info[2]}-head which is a= {info[-1]}"

    if len(info) == 5:
        if feature.startswith("headrelrel"):  # f'headrelrel_{pos}_{relation}_{head}_{headrelation.lower()}'
            info_rel, rel_link = get_relation(info[-1])
            info[-1] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            info_pos, pos_link = get_relation(info[1])
            info[1] = info_pos

            info_pos, pos_link = get_relation(info[2])
            info[1] = info_pos

            info_rel, rel_link = get_relation(info[3])
            info[3] = info_rel

            new_features = (
                f"{info[1]} is a {info[2]} of head {info[3]} where that {info[3]} is the= {info[-1]}"
            )

        elif feature.startswith("headfeat"):  # f'headfeat_{head_pos}_{relation}_{feat}_{value}'
            info_rel, rel_link = get_relation(info[2])
            info[2] = info_rel

            info_pos, pos_link = get_relation(info[1])
            info[1] = info_pos

            new_features = f"{dependent}  is a {info[2]}  of {info[1]} with {info[3]}= {info[4]}"

        elif feature.startswith("depfeat"):  # f'dpefeat_{pos}_{relation}_{feat}_{value}'
            info_rel, rel_link = get_relation(info[2])
            info[2] = info_rel

            info_pos, pos_link = get_relation(info[1])
            info[1] = info_pos

            new_features = f"{info[1]} is a {info[2]} with {info[3]}= {info[4]}"

    return new_features
