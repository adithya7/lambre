import pyconll
import random
from copy import deepcopy
import argparse
from collections import defaultdict


def read_ud_um():
    with open("ud-um.tsv") as inp:
        lines = inp.readlines()
    um2ud = {}
    ud2um = {}
    for l in lines:
        l = l.strip().split("\t")
        if len(l) > 1:
            if l[0] and l[1]:
                ud2um[l[0]] = l[1]
                um2ud[l[1]] = l[0]
    return um2ud, ud2um


def read_um(f):
    with open(f) as inp:
        lines = inp.readlines()
    d = {}
    for l in lines:
        l = l.strip()
        if l:
            l = l.split("\t")
            lemma = l[0]
            form = l[1]
            tags = l[2]
            if lemma not in d:
                d[lemma] = {}
            if form in d[lemma] and form != "-":
                d[lemma][form].append(tags)
            elif form != "—":
                d[lemma][form] = [tags]
    return d


def convert_um2ud(umf, um2ud):
    umf = umf[0].split(";")
    d = {}
    pos = ""
    for feat in umf:
        if feat in um2ud:
            temp = um2ud[feat]
            if "=" in temp:
                temp = temp.split("=")
                if temp[0] not in d:
                    d[temp[0]] = set()
                d[temp[0]].add(temp[1])
            else:
                pos = temp
    return pos, d


def count_changes(source, targets):
    counts = []
    for t in targets:
        c = 0
        for tf in t:
            if tf in source:
                if source[tf] != t[tf]:
                    c += 1
        counts.append(c)
    return counts


def sample_noise(data, um, um2ud, k=1):
    new_sents = []
    total_count, altered_count = 0, 0
    alt_feat_dict = defaultdict(int)
    skipped_count = 0
    for sentence in data:
        try:
            candidate_sents = []
            for _, token in enumerate(sentence):
                token_id = token.id
                feats = token.feats
                upos = token.upos
                lemma = token.lemma
                form = token.form
                if lemma in um:
                    # Convert UD feats to UM
                    if form in um[lemma]:
                        options = list(um[lemma].keys())
                        options.remove(form)
                        # TODO: Allow for multi-word insertions
                        options = [o for o in options if " " not in o]
                        option_feats = [convert_um2ud(um[lemma][o], um2ud)[1] for o in options]
                        counts = count_changes(feats, option_feats)
                        options = [o for i, o in enumerate(options) if counts[i] == k]
                        counts = [c for i, c in enumerate(counts) if c == k]

                        # for choice in options:
                        if options:
                            choice = random.choice(options)

                            new_sent = deepcopy(sentence)
                            new_feats = new_sent[token_id].feats
                            option_feats = convert_um2ud(um[lemma][choice], um2ud)[1]
                            old_feat = ""
                            for option_feat in option_feats:
                                if option_feat in feats:
                                    if feats[option_feat] != option_feats[option_feat]:
                                        old_feats = feats[option_feat]
                                        old_feat = option_feat
                                        new_feats[option_feat] = option_feats[option_feat]

                            new_token_string = new_sent[token_id].conll()
                            new_token_string = new_token_string.split("\t")
                            new_token_string = (
                                new_token_string[:1]
                                + [choice]
                                + new_token_string[2:-1]
                                + [f"{new_token_string[-1]}|modified:{old_feat}={','.join(list(old_feats))}"]
                            )
                            new_token_string = "\t".join(new_token_string)
                            modified_token = pyconll.unit.token.Token(new_token_string).conll()

                            new_str = []
                            for token in new_sent:
                                if token.id == token_id:
                                    new_str.append(modified_token)
                                else:
                                    new_str.append(token.conll())

                            candidate_sents.append((new_str, upos, old_feat))

            """ randomly select one altered sentence for each input sentence """
            if len(candidate_sents) > 0:
                altered_count += 1
                new_sent, upos, old_feat = random.choice(candidate_sents)
                alt_feat_dict[(upos, old_feat)] += 1
            else:
                new_sent = [token.conll() for token in sentence]
            new_sents.append(new_sent)

            # # """ add all possible noisy sentences """
            # sents_toadd = []
            # if len(candidate_sents) > 0:
            #     altered_count += 1
            #     for new_sent, upos, old_feat in candidate_sents:
            #         alt_feat_dict[(upos, old_feat)] += 1
            #         sents_toadd.append(new_sent)
            # else:
            #     new_sent = [token.conll() for token in sentence]
            #     sents_toadd.append(new_sent)
            # new_sents.extend(sents_toadd)

            total_count += 1
        except:
            skipped_count += 1
            continue

    print("altered fraction: %.2f" % (float(altered_count) * 100 / total_count))
    print("skipped count: %d" % (skipped_count))
    return new_sents, alt_feat_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate noisy outputs")
    parser.add_argument("-unimorph", type=str, help="unimorph file for lg")
    parser.add_argument("-orig", type=str, help="original conllu")
    parser.add_argument("-alt", type=str, help="altered conllu")
    args = parser.parse_args()

    treebank = args.orig
    new_treebank = args.alt

    data = pyconll.load_from_file(treebank)

    um = read_um(args.unimorph)
    um2ud, ud2um = read_ud_um()

    random.seed(23)
    new_sents, alt_feat_dict = sample_noise(data, um, um2ud)
    with open(new_treebank, "w") as op:
        for new_sent in new_sents:
            sent_ = pyconll.load_from_string("\n".join(new_sent))
            assert len(sent_) == 1, "more than one sentence"
            # op.write("# text = " + " ".join([t_.form for t_ in sent_[0] if t_.form]) + "\n")
            op.write("\n".join(new_sent) + "\n\n")
    alt_feat_dict = sorted(alt_feat_dict.items(), key=lambda x: x[0])
    print(args.unimorph.split("/")[-1])
    for (pos, feat), count in alt_feat_dict:
        print("%s:%s\t%.2f" % (pos, feat, count * 100 / len(new_sents)))
