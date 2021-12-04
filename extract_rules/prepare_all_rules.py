import argparse, os


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='/Users/aditichaudhary/Documents/CMU/argument_structure/syntax_lex')
parser.add_argument("--lang", type=str, nargs='+',default=['tr_imst'])
parser.add_argument("--output", type=str, default="../rules/all_rules.txt")

args = parser.parse_args()



def validLabel(label):
    if label == 'chance-agree' or label == 'NA':
        return False
    return True

def readRules(input, rules):
    # req-agree
    # active: deprel_subj
    # non_active: headpos_NOUN
    # agree:2287, disagree: 28, total: 2315

    with open(input, 'r') as fin:
        lines = fin.readlines()
        line_num = 0

        filename = os.path.basename(input)
        filename = filename.split("_")
        lang_id = filename[0]
        feature = filename[2]

        task = input.split("/")[-2].lower()
        while line_num < len(lines):
            leaf_label = lines[line_num].lstrip().rstrip()
            line_num += 1
            active = lines[line_num].lstrip().rstrip().split("active: ")
            if len(active) == 2:
                active = active[1]
            else:
                active = "NA"
            line_num += 1
            non_active = lines[line_num].lstrip().rstrip().split("non_active: ")
            if len(non_active) == 2:
                non_active = non_active[1]
            else:
                non_active = "NA"

            line_num += 1
            data = lines[line_num].lstrip().rstrip().split("total: ")[1]
            line_num +=2

            if validLabel(leaf_label):
                rules.append((task, feature, lang_id, leaf_label, active, non_active, data))
        return rules
if __name__ == "__main__":
    rules = []
    input_files = []
    for lang in args.lang:
        task = 'Agreement'
        input_dir = f'{args.input}/{lang}/{task}'
        for feature in ['Gender', 'Person', 'Number']:
            input_file = f'{input_dir}/{lang}_{feature}_rules.txt'
            if os.path.exists(input_file):
                input_files.append(input_file)

        task =  'WordOrder'
        input_dir = f'{args.input}/{lang}/{task}'
        for feature in ['subject-verb', 'object-verb', 'adjective-noun', 'noun-adposition', 'numeral-noun']:
            input_file = f'{input_dir}/{lang}_{feature}_rules.txt'
            if os.path.exists(input_file):
                input_files.append(input_file)

        task = 'CaseMarking'
        input_dir = f'{args.input}/{lang}/{task}'
        for feature in ['DET', 'NOUN', 'VERB', 'ADJ', 'PROPN', 'PRON', 'ADV', 'NUM']:
            input_file = f'{input_dir}/{lang}_{feature}_rules.txt'
            if os.path.exists(input_file):
                input_files.append(input_file)


    for input in input_files:
        print('Processing, ', input)
        readRules(input, rules)
    print(len(rules))

    with open(args.output, 'w') as fout:
        for (task, feature, lang_id, leaf_label, active, non_active, data) in rules:
            fout.write(task + "\t" + feature + "\t" + lang_id + "\t" + leaf_label + "\t" + active + "\t" + non_active + "\t" + data + "\n")