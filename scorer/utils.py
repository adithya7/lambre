"""
utils for metric
"""
from collections import defaultdict
from copy import deepcopy

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


def load_rule_file(INP_FILE):

    lang2rule = {}
    with open(INP_FILE, "r") as rf:
        #Agreement	Number	tr_imst	req-agree	headmatch_True ### deppos_PROPN	NA	454
        lang = ""
        for line in rf:
            line_txt = line.strip().split("\t")
            task, model, lang, label, active, non_active, total_samples = line_txt[0], line_txt[1], line_txt[2], line_txt[3], line_txt[4], line_txt[5], line_txt[6]
            if lang not in lang2rule:
                lang2rule[lang] = {}
            if task not in lang2rule[lang]:
                lang2rule[lang][task] = defaultdict(list)

            lang2rule[lang][task][model].append((label, active, non_active, total_samples))


    return lang2rule


def isPropertyPresent(prop, token, isHead=False):
    if token is None:
        return False

    if not isHead:#Checking if the head of the the dependent is no root
        if token.head == '0':
            return False

    if prop in token.feats:
        return True

    return False


def extractFeaturesFromRules(rules):
    active_features, nonactive_features, labels = [],[],[]

    for (label, active, non_active, _) in rules:
        active = active.split(" ### ")
        non_active = non_active.split(" ### ")

        one_active, one_nonactive = [],[]
        for f in active:
            if f != 'NA':
                one_active.append(f)

        for f in non_active:
            if f != 'NA':
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


    feature = f'deppos_{pos}'
    features.append(feature)

    if token.deprel:
        relation = token.deprel.lower()
        feature = f'deprel_{relation}'
        features.append(feature)

    for feat in feats:
        feature = f'depfeat_{feat}_'
        value = getFeatureValue(feat, feats)
        feature += f'{value}'
        features.append(feature)

    if use_lexical:
        lemma = isValidLemma(lemma, pos)
        if lemma:
            feature = f'lemma_{lemma}'
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
                    feature = f'neighborhood_{lemma}'
                    features.append(feature)

    if token.head != '0' and token.head is not None:
        head_pos = sentence[token.head].upos
        headrelation = sentence[token.head].deprel
        head_feats = sentence[token.head].feats
        headhead = sentence[token.head].head
        head_lemma = sentence[token.head].lemma

        feature = f'headpos_{head_pos}'
        features.append(feature)

        if headrelation and headrelation != 'root' and headrelation != 'punct':
            feature = f'headrelrel_{headrelation.lower()}'
            features.append(feature)

            feature = f'headrelrel_{head_pos}_{relation}_{headrelation.lower()}'
            features.append(feature)

            feature = f'headrelrel_{head_pos}_{headrelation.lower()}'
            features.append(feature)

        for feat in head_feats:  # Adding features for dependent token (maybe more commonly occurring)
            feature = f'headfeat_{head_pos}_{feat}_'
            value = getFeatureValue(feat, head_feats)
            feature += f'{value}'
            features.append(feature)

            feature = f'headfeat_{feat}_{value}'
            features.append(feature)

            feature = f'headfeat_{head_pos}_{relation}_{feat}_'
            value = getFeatureValue(feat, head_feats)
            feature += f'{value}'
            features.append(feature)

            feature = f'headfeat_{head_pos}_{feat}_{value}'
            features.append(feature)

            feature = f'headfeatrel_{relation}_{feat}_{value}'
            features.append(feature)

        if headhead and headhead != '0':

            headhead_feats = sentence[headhead].feats
            for prop in ['Gender', 'Person', 'Number']:
                headlabel = getFeatureValue(prop, head_feats)
                if headlabel and prop in headhead_feats:
                    headhead_value = getFeatureValue(prop, headhead_feats)
                    if headlabel == headhead_value:
                        feature = f'headmatch_True_{prop}'
                        features.append(feature)

            headheadheadlemma = isValidLemma(sentence[headhead].lemma,
                                                  sentence[headhead].upos)
            if use_lexical and headheadheadlemma:
                feature = f'headheadlemma_{headheadheadlemma}'
                features.append(feature)

        if use_lexical:
            head_lemma = isValidLemma(head_lemma, head_pos)
            if head_lemma:
                feature = f'headlemma_{head_lemma}'
                features.append(feature)

        if 'Case' in head_feats and 'Case' in feats:
            label = getFeatureValue('Case', feats)
            headlabel = getFeatureValue('Case', head_feats)

            if label == headlabel:  # If agreement between the head-dep
                feature = f'agreepos_{head_pos}'
                features.append(feature)

                feature = f'agreerel_{relation}'
                features.append(feature)

                if headrelation and headrelation != 'root' and headrelation != 'punct':
                    feature = f'agree_{relation}_{head_pos}_{headrelation.lower()}'
                    features.append(feature)

                    feature = f'agree_{headrelation.lower()}'
                    features.append(feature)


    # get other dep tokens of the head
    dep = dep_data_token.get(token.head, [])
    for d in dep:
        if d == token.id:
            continue
        depdeprelation = sentence[d].deprel
        if depdeprelation and depdeprelation != 'punct':
            feature = f'depheadrel_{depdeprelation}'
            features.append(feature)

        depdeppos = sentence[d].upos
        feature = f'depheadpos_{depdeppos}'
        features.append(feature)

        depdeplemma = isValidLemma(sentence[d].lemma, sentence[d].upos)
        if use_lexical and depdeplemma:
            feature = f'depheadlemma_{depdeplemma}'
            features.append(feature)

    # adding the children of the dep token
    for dep in dep_data_token[token.id]:
        deptoken = sentence[dep]
        feature = f'depdeppos_{deptoken.upos}'
        features.append(feature)

        deprel = deptoken.deprel
        if deprel and deprel != 'root' and deprel != 'punct':
            feature = f'depdeprel_{deprel}'
            features.append(feature)

        deplemma = isValidLemma(deptoken.lemma, deptoken.upos)
        if use_lexical and deplemma:
            feature = f'depdeplemma_{deplemma}'
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
    if upos in ['PUNCT', 'NUM', 'PROPN' 'X', 'SYM']:
        return None
    if lemma:
        lemma = lemma.lower()
        lemma = lemma.replace("\"", "").replace("\'", "")
        if lemma == "" or lemma == " ":
            return None
        else:
            return lemma
    return None


def checkModelApplicable(task, model, token, sent):
    if task == 'agreement':
        # If the model (e.g. gender,person, number) in the head and dep, only then check for agreement match
        if isPropertyPresent(model, token) and isPropertyPresent(model, sent[token.head], isHead=True):
            dep_value = getFeatureValue(model, token.feats)
            head_value = getFeatureValue(model, sent[token.head].feats)

            if dep_value == head_value: #observed agreement in the example
                return 1
            else:
                return 0
        else:
            return -1

    elif task == 'wordorder':
        # the model is subject-verb for example
        pos = token.upos
        if token.head == '0' or not token.head:
            return -1
        head_pos = sent[token.head].upos
        relation = token.deprel

        wals_features = {'subject-verb': ['subj_VERB'],
                         'object-verb': ['obj_VERB'],
                         'noun-adposition': ['NOUN_ADP', 'PRON_ADP', 'PROPN_ADP'],  # When adp is the syntactic head
                         'adjective-noun': ['ADJ_mod_NOUN', 'ADJ_mod_PROPN', 'ADJ_mod_PRON'],
                         'numeral-noun': ['NUM_mod_NOUN', 'NUM_mod_PROPN', 'NUM_mod_PRON'],
                         }
        defined_features = wals_features[model]
        isValid = False
        #Check if the model is applicable for this datapoint, e.g. for subject-verb check if the dep is a subj and its head is a verb
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
            #Get the observed label
            id2index = sent._ids_to_indexes
            token_position = id2index[token.id]
            head_position = id2index[token.head]
            if token_position < head_position:
                label = 'before'
            else:
                label = 'after'
            return label
        else:
            return -1

    elif task == 'casemarking':
        # the model is casemarking for nouns for example
        pos = token.upos
        feats = token.feats
        if model != pos or 'Case' not in feats:
            return -1
        label = getFeatureValue('Case', feats)
        return label

def isGrammarRuleApplicable(featuresInDatapoint, one_rule_active, one_rule_nonactive, prop=None):
    updated_featuresInDatapoint = []
    for f in featuresInDatapoint:
        if f.startswith('headmatch_True') and prop:
            fprop = f.split("_")[1]
            if prop == fprop:
                f = 'headmatch_True'
            else:
                f = None

        if f:
            updated_featuresInDatapoint.append(f)

    valid = True
    #check if the active features defined in the rule are active for this datapoint
    for f in one_rule_active:
        if f and f not in updated_featuresInDatapoint:
            valid = False
            break

    if not valid:
        return False

    #check if any of the nonactive features in the rule are really not present for this datapoint
    for f in one_rule_nonactive:
        if f and f in updated_featuresInDatapoint:
            valid = False
            break

    return valid


def printExamples(rules_not_followed, sent_tokens, token, token_num, sent, id2index, fout, task):
    if len(rules_not_followed) > 0:  # there are rules which were not followed for this datapoint
        if task == 'agreement':
            for model, (one_active, one_nonactive, label) in rules_not_followed.items():
                sent_example_tokens = deepcopy(sent_tokens)
                token_feature_value = getFeatureValue(model, token.feats)
                sent_example_tokens[token_num] = "***" + sent_example_tokens[token_num] + f"({model}={token_feature_value})***"

                token_head_num = id2index[token.head]
                headtoken_feature_value = getFeatureValue(model, sent[token.head].feats)
                sent_example_tokens[token_head_num] = "***" + sent_example_tokens[
                    token_head_num] + f"({model}={headtoken_feature_value})***"

                fout.write(f'Example: {" ".join(sent_example_tokens)}\n')
                fout.write(
                    f'{model} agreement not followed by tokens marked *** because following rule was not satisfied:\n')
                if len(one_active) > 0:
                    fout.write(f'Required Active features in the rule: {" ### ".join(one_active)}\n')
                if len(one_nonactive) > 0:
                    fout.write(f'Required Non-active features in that rule: {" ### ".join(one_nonactive)}\n')
                fout.write("\n")

        elif task == 'wordorder':
            for model, (one_active, one_nonactive, label) in rules_not_followed.items():
                dep, head = model.split("-")[0], model.split("-")[1]
                if head == 'noun':
                    head = 'nominal'
                if label == 'before':
                    not_label = 'after'
                else:
                    not_label = 'before'

                sent_example_tokens = deepcopy(sent_tokens)
                sent_example_tokens[token_num] = "***" + sent_example_tokens[token_num] + f"({dep})***"

                token_head_num = id2index[token.head]
                sent_example_tokens[token_head_num] = "***" + sent_example_tokens[token_head_num] + f"({head})***"

                fout.write(f'Example: {" ".join(sent_example_tokens)}\n')
                fout.write(
                    f'{model} order not followed for tokens marked ***, predicted order is {label} but observed is {not_label}. because following rule was not satisfied:\n')
                if len(one_active) > 0:
                    fout.write(f'Required Active features in the rule: {" ### ".join(one_active)}\n')
                if len(one_nonactive) > 0:
                    fout.write(f'Required Non-active features in that rule: {" ### ".join(one_nonactive)}\n')
                fout.write("\n")

        elif task == 'casemarking':
            for model, (one_active, one_nonactive, label) in rules_not_followed.items():
                sent_example_tokens = deepcopy(sent_tokens)
                value = getFeatureValue('Case', token.feats)
                sent_example_tokens[token_num] = "***" + sent_example_tokens[token_num] + f"({token.upos}'s Case={value})***"

                fout.write(f'Example: {" ".join(sent_example_tokens)}\n')
                fout.write(
                    f'{model} case marking not followed for tokens marked ***, predicted case is [{label}] but observed is [{value}], because following rule was not satisfied:\n')
                if len(one_active) > 0:
                    fout.write(f'Required Active features in the rule: {" ### ".join(one_active)}\n')
                if len(one_nonactive) > 0:
                    fout.write(f'Required Non-active features in that rule: {" ### ".join(one_nonactive)}\n')
                fout.write("\n")