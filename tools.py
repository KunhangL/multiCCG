from typing import List
import json

import ccg_rules
from base import ConstituentNode, Token, Category


to_X_features = {
    "S/(S\\NP)": "(S[X]/(S[X]\\NP))",
    "(S\\NP)\\((S\\NP)/NP)": "((S[X]\\NP)\\((S[X]\\NP)/NP))",
    "(S\\NP)\\((S\\NP)/PP)": "((S[X]\\NP)\\((S[X]\\NP)/PP))",
    "((S\\NP)/NP)\\(((S\\NP)/NP)/NP)": "(((S[X]\\NP)/NP)\\(((S[X]\\NP)/NP)/NP))",
    "((S\\NP)/PP)\\(((S\\NP)/PP)/NP)": "(((S[X]\\NP)/PP)\\(((S[X]\\NP)/PP)/NP))"
}


def apply_binary_rules_to_categories(categories: List[str]):
    # apply binary rules to all possible category pairs
    # and collect instantiated rules along with all possible results
    instantiated_rules = list()
    possible_results = set()

    progress = 0
    for i in range(len(categories)):
        for j in range(len(categories)):

            progress += 1
            if progress % len(categories) == 0:
                print(
                    f'progress: {progress} / {len(categories) * len(categories)}'
                )

            constituent_1 = ConstituentNode(tag=Category.parse(categories[i]))
            constituent_2 = ConstituentNode(tag=Category.parse(categories[j]))
            results = [categories[i], categories[j], []]
            for binary_rule in ccg_rules.binary_rules:
                result = binary_rule(constituent_1, constituent_2)
                if result:
                    results[2].append(
                        [
                            str(result.tag),
                            ccg_rules.abbreviated_rule_name[binary_rule.__name__]
                        ]
                    )
                    possible_results.add(str(result.tag))
            if results[2]:
                instantiated_rules.append(results)

    return instantiated_rules, possible_results


def collect_unary_rules(data_dir: str, saving_dir: str):
    # apply unary rules to collected instantiated unary rules.
    # so as to save the results with the names of applied rules
    with open(data_dir, 'r', encoding='utf8') as f:
        seen_unary_rules = json.load(f)

    instantiated_unary_rules = list()

    i = 0
    for unary_rule in seen_unary_rules:
        i += 1
        print(f'progress {i} / {len(seen_unary_rules)}')
        instantiated_rule = [unary_rule[0], unary_rule[1]]
        tag_before = str(Category.parse(unary_rule[0]))
        tag_after = str(Category.parse(unary_rule[1]))
        if not ccg_rules._is_type_raised(
            Category.parse(unary_rule[1])
        ):
            instantiated_rule.append('TC')
        else:
            type = Category.parse(unary_rule[1]).left
            for rule in ccg_rules.unary_rules:
                result = rule(
                    x=ConstituentNode(tag=Category.parse(tag_before)),
                    T=type
                )
                if str(result.tag) == tag_after:
                    instantiated_rule.append(ccg_rules.abbreviated_rule_name[rule.__name__])
                    matched = True

        instantiated_unary_rules.append(instantiated_rule)

    print('Number of instantiated unary rules: ', len(instantiated_unary_rules))
    with open(saving_dir, 'w', encoding='utf8') as f:
        json.dump(instantiated_unary_rules, f, indent=2, ensure_ascii=False)


def collect_binary_rules(data_dir: str, saving_dir: str):
    # collect seen binary rules (a list of category string pairs),
    # apply binary rules to them,
    # and save the seen pairs along with results
    with open(data_dir, 'r', encoding='utf8') as f:
        seen_binary_rules = json.load(f)

    instantiated_binary_rules = list()

    i = 0
    for binary_rule in seen_binary_rules:
        i += 1
        print(f'progress {i} / {len(seen_binary_rules)}')
        results = [binary_rule[0], binary_rule[1], []]
        tag_0 = str(Category.parse(binary_rule[0]))
        tag_1 = str(Category.parse(binary_rule[1]))
        tag_0 = to_X_features[tag_0] if tag_0 in to_X_features.keys() else tag_0
        tag_1 = to_X_features[tag_1] if tag_1 in to_X_features.keys() else tag_1
        for rule in ccg_rules.binary_rules:
            result = rule(
                ConstituentNode(tag=Category.parse(tag_0)),
                ConstituentNode(tag=Category.parse(tag_1))
            )
            if result:
                to_add = [
                    str(result.tag),
                    ccg_rules.abbreviated_rule_name[rule.__name__]
                ]
                if to_add not in results[2]:
                    if Category.parse(to_add[0]) not in [
                        Category.parse(item[0]) for item in results[2]
                    ]:
                        results[2].append(to_add)
        if results[2]:
            if results[:2] not in [
                rule[:2] for rule in instantiated_binary_rules
            ]:
                instantiated_binary_rules.append(results)
            else:
                idx = [
                    rule[:2]for rule in instantiated_binary_rules
                ].index(results[:2])
                instantiated_binary_rules[idx][2].extend(results[2])

    print('Number of instantiated binary rules: ', len(instantiated_binary_rules))
    with open(saving_dir, 'w', encoding='utf8') as f:
        json.dump(instantiated_binary_rules, f, indent=2, ensure_ascii=False)


def collect_cats_from_markedup(file_dir: str) -> List[str]:
    cats = list()
    with open(file_dir, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        if line[0] not in ['=', '#', ' ', '\n']:
            cats.append(line.strip())
    return cats


def to_auto(node: ConstituentNode) -> str:
    # convert one ConstituentNode to an .auto string
    if len(node.children) == 1 and isinstance(node.children[0], Token):
        token = node.children[0]
        cat = token.tag
        word = denormalize(token.contents)
        pos = token.POS
        return f'(<L {cat} {pos} {pos} {word} {cat}>)'
    else:
        cat = node.tag
        children = ' '.join(to_auto(child) for child in node.children)
        num_children = len(node.children)
        head_is_left = 0 if node.head_is_left else 1
        return f'(<T {cat} {head_is_left} {num_children}> {children} )'


# source: https://github.com/masashi-y/depccg
def normalize(word: str) -> str:
    if word == "-LRB-":
        return "("
    elif word == "-RRB-":
        return ")"
    elif word == "-LCB-":
        return "{"
    elif word == "-RCB-":
        return "}"
    elif word == "-LSB-":
        return "["
    elif word == "-RSB-":
        return "]"
    else:
        return word


# source: https://github.com/masashi-y/depccg
def denormalize(word: str) -> str:
    if word == "(":
        return "-LRB-"
    elif word == ")":
        return "-RRB-"
    elif word == "{":
        return "-LCB-"
    elif word == "}":
        return "-RCB-"
    elif word == "[":
        return "-LSB-"
    elif word == "]":
        return "-RSB-"
    word = word.replace(">", "-RAB-")
    word = word.replace("<", "-LAB-")
    return word


def build_category2idx(folder_path: str, result_path: str) -> None:
    # Build one category2idx dictionary after applying frequency cutoff, for Anh's treebanks.
    import os
    from os.path import join as pjoin

    category2idx = dict()

    folder_names = os.listdir(folder_path)
    for folder_name in folder_names:
        file_names = os.listdir(pjoin(folder_path, folder_name))
        for file_name in file_names:
            if file_name.endswith('train.lexicon'):
                with open(pjoin(folder_path, folder_name, file_name), 'r', encoding='utf8') as rf:
                    lines = rf.readlines()
                    for line in lines:
                        items = [item.strip() for item in line.split('\t') if item != '']
                        token, category, frequency = items[0], items[1], int(items[2])
                        if category not in category2idx.keys():
                            category2idx[category] = frequency
                        else:
                            category2idx[category] += frequency
    print(len(category2idx))
    category2idx = {k:v for k,v in category2idx.items() if v>=10}
    cnt = 0
    for k in category2idx:
        category2idx[k] = cnt
        cnt += 1

    with open(result_path, 'w', encoding='utf8') as wf:
        json.dump(category2idx, wf, indent=2, ensure_ascii=False)


def merge_auto_files(folder_path: str, mode: str, result_folder: str) -> None:
    # Merge all auto files from the treebanks, mode: ['train', 'test', 'dev']
    import os
    from os.path import join as pjoin

    data = []
    folder_names = os.listdir(folder_path)
    for folder_name in folder_names:
        file_names = os.listdir(pjoin(folder_path, folder_name))
        for file_name in file_names:
            if file_name.endswith(mode + '.auto'):
                with open(pjoin(folder_path, folder_name, file_name), 'r', encoding='utf8') as rf:
                    data.extend(rf.readlines())

    with open(pjoin(result_folder, 'treebanks_' + mode + '.auto'), 'w', encoding='utf8') as wf:
        wf.writelines(data)


def list_auto_files(folder_path: str, mode: str) -> None:
    # List paths to auto files from the treebanks, mode: ['train', 'test', 'dev']
    import os
    from os.path import join as pjoin

    path_names = []
    folder_names = os.listdir(folder_path)
    for folder_name in folder_names:
        file_names = os.listdir(pjoin(folder_path, folder_name))
        for file_name in file_names:
            if file_name.endswith(mode + '.auto'):
                path_names.append(pjoin(folder_path, folder_name, file_name))
    print(*path_names, sep=',\n')
    print(len(path_names))


if __name__ == '__main__':
    # sample use
    collect_binary_rules(data_dir='./data/instantiated_binary_rules_raw.json',
                         saving_dir='./data/instantiated_binary_rules.json')
    collect_unary_rules(data_dir='./data/instantiated_unary_rules_raw.json',
                        saving_dir='./data/instantiated_unary_rules.json')

    # build_category2idx(folder_path='./treebanks', result_path='./data/treebanks_new_lexical_category2idx_cutoff.json')

    # merge_auto_files(folder_path='./treebanks', mode='test', result_folder='./data')

    # list_auto_files(folder_path='./treebanks_new_subset', mode='train')