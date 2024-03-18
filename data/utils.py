"""
Some functions used to collect information from designated data
"""

import sys
import json
from typing import List, Dict

sys.path.append('..')
from base import ConstituentNode
from data_loader import DataItem, load_auto_file


def collect_unary_rules(data_items: List[DataItem], saving_dir: str) -> None:
    """
    Input:
        data_items - a list of DataItems
        saving_dir - the directory used to store collected unary rules
                     sorted by number they appear in
    """
    def _iter(node, unary_rules):
        if isinstance(node, ConstituentNode):
            if len(node.children) == 1:
                child_cat = str(node.children[0].tag)
                parent_cat = str(node.tag)
                if child_cat not in unary_rules.keys():
                    unary_rules[child_cat] = dict()
                if parent_cat not in unary_rules[child_cat].keys():
                    unary_rules[child_cat][parent_cat] = 0
                unary_rules[child_cat][parent_cat] += 1
            for child in node.children:
                _iter(child, unary_rules)

    unary_rules = dict()
    for item in data_items:
        _iter(item.tree_root, unary_rules)

    for child_cat in unary_rules:
        unary_rules[child_cat] = [
            [parent_cat, cnt]
            for parent_cat, cnt in unary_rules[child_cat].items()
        ]
        unary_rules[child_cat] = sorted(unary_rules[child_cat], key=lambda x: x[1], reverse=True)

    unary_rules_new = list()
    for child_cat in unary_rules.keys():
        for item in unary_rules[child_cat]:
            unary_rules_new.append([child_cat, item[0], item[1]])
    unary_rules_new = sorted(unary_rules_new, key=lambda x: x[2], reverse=True)

    # Delete no-change rules
    cleaned_unary_rules = list()
    for rule in unary_rules_new:
        if rule[0] != rule[1]:
            cleaned_unary_rules.append(rule)

    with open(saving_dir, 'w', encoding='utf8') as f:
        json.dump(cleaned_unary_rules, f, indent=2, ensure_ascii=False)


def collect_binary_rules(data_items: List[DataItem], saving_dir: str):
    """
    Input:
        data_items - a list of DataItems
        saving_dir - the directory used to store collected binary rules
                     sorted by number they appear in
    """
    def _iter(node, binary_rules):
        if isinstance(node, ConstituentNode):
            if len(node.children) == 2:
                child_cat_0 = str(node.children[0].tag)
                child_cat_1 = str(node.children[1].tag)
                parent_cat = str(node.tag)
                if child_cat_0 not in binary_rules:
                    binary_rules[child_cat_0] = dict()
                if child_cat_1 not in binary_rules[child_cat_0]:
                    binary_rules[child_cat_0][child_cat_1] = dict()
                if parent_cat not in binary_rules[child_cat_0][child_cat_1]:
                    binary_rules[child_cat_0][child_cat_1][parent_cat] = 0
                binary_rules[child_cat_0][child_cat_1][parent_cat] += 1
            for child in node.children:
                _iter(child, binary_rules)

    binary_rules = dict()
    for data_item in data_items:
        _iter(data_item.tree_root, binary_rules)

    binary_rules_new = []
    for left in binary_rules:
        for right in binary_rules[left]:
            for result in binary_rules[left][right]:
                binary_rules_new.append(
                    [left, right, result, binary_rules[left][right][result]]
                )

    binary_rules_new = sorted(binary_rules_new, key=lambda x: x[3], reverse=True)

    # Deduplicate repetitive results
    deduplicated_binary_rules = list()
    for rule in binary_rules_new:
        new_rule = list()
        new_rule.append(rule[0])
        new_rule.append(rule[1])
        new_rule.append([])
        for result in rule[2]:
            if result not in new_rule[2]:
                new_rule[2].append(result)
        deduplicated_binary_rules.append(new_rule)

    with open(saving_dir, 'w', encoding='utf8') as f:
        json.dump(deduplicated_binary_rules, f, indent=2, ensure_ascii=False)


def collect_roots(data_items: List[DataItem]) -> Dict[str, int]:
    """
    Input:
        data_items - a list of DataItems
    Output:
        A dictionary storing all possible root categories
        along with their appearance numbers
    """
    root_set = dict()
    for item in data_items:
        tag = str(item.tree_root.tag)
        if tag not in root_set:
            root_set[tag] = 0
        root_set[tag] += 1
    
    return root_set


if __name__ == '__main__':

    # Sample use to generate raw instantiated rules

    def _load(data_paths):
        all_data_items = list()
        for path in data_paths:
            try:
                data_items, _ = load_auto_file(path)
                all_data_items.extend(data_items)
            except:
                continue
        return all_data_items

    treebanks_train_data_paths = [
        '../treebanks/UD_Portuguese-GSD/pt_gsd-ud-train.auto',
        '../treebanks/UD_Vietnamese-VTB/vi_vtb-ud-train.auto',
        '../treebanks/UD_Telugu-MTG/te_mtg-ud-train.auto',
        '../treebanks/UD_Wolof-WTB/wo_wtb-ud-train.auto',
        '../treebanks/UD_Catalan-AnCora/ca_ancora-ud-train.auto',
        '../treebanks/UD_Turkish-IMST/tr_imst-ud-train.auto',
        '../treebanks/UD_Arabic-PADT/ar_padt-ud-train.auto',
        '../treebanks/UD_Croatian-SET/hr_set-ud-train.auto',
        '../treebanks/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.auto',
        '../treebanks/UD_Marathi-UFAL/mr_ufal-ud-train.auto',
        '../treebanks/UD_Turkish-FrameNet/tr_framenet-ud-train.auto',
        '../treebanks/UD_Western_Armenian-ArmTDP/hyw_armtdp-ud-train.auto',
        '../treebanks/UD_Korean-Kaist/ko_kaist-ud-train.auto',
        '../treebanks/UD_Czech-CLTT/cs_cltt-ud-train.auto',
        '../treebanks/UD_Welsh-CCG/cy_ccg-ud-train.auto',
        '../treebanks/UD_Italian-ParTUT/it_partut-ud-train.auto',
        '../treebanks/UD_Finnish-FTB/fi_ftb-ud-train.auto',
        '../treebanks/UD_Hungarian-Szeged/hu_szeged-ud-train.auto',
        '../treebanks/UD_French-Sequoia/fr_sequoia-ud-train.auto',
        '../treebanks/UD_English-ParTUT/en_partut-ud-train.auto',
        '../treebanks/UD_Turkish-Kenet/tr_kenet-ud-train.auto',
        '../treebanks/UD_Irish-IDT/ga_idt-ud-train.auto',
        '../treebanks/UD_Belarusian-HSE/be_hse-ud-train.auto',
        '../treebanks/UD_Bulgarian-BTB/bg_btb-ud-train.auto',
        '../treebanks/UD_Old_East_Slavic-TOROT/orv_torot-ud-train.auto',
        '../treebanks/UD_Dutch-LassySmall/nl_lassysmall-ud-train.auto',
        '../treebanks/UD_German-HDT/de_hdt-ud-train.auto',
        '../treebanks/UD_Tamil-TTB/ta_ttb-ud-train.auto',
        '../treebanks/UD_Maltese-MUDT/mt_mudt-ud-train.auto',
        '../treebanks/UD_Icelandic-Modern/is_modern-ud-train.auto',
        '../treebanks/UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.auto',
        '../treebanks/UD_Polish-PDB/pl_pdb-ud-train.auto',
        '../treebanks/UD_Latin-ITTB/la_ittb-ud-train.auto',
        '../treebanks/UD_Ukrainian-IU/uk_iu-ud-train.auto',
        '../treebanks/UD_Galician-CTG/gl_ctg-ud-train.auto',
        '../treebanks/UD_Uyghur-UDT/ug_udt-ud-train.auto',
        '../treebanks/UD_Faroese-FarPaHC/fo_farpahc-ud-train.auto',
        '../treebanks/UD_Latin-LLCT/la_llct-ud-train.auto',
        '../treebanks/UD_Latvian-LVTB/lv_lvtb-ud-train.auto',
        '../treebanks/UD_English-LinES/en_lines-ud-train.auto',
        '../treebanks/UD_Danish-DDT/da_ddt-ud-train.auto',
        '../treebanks/UD_English-EWT/en_ewt-ud-train.auto',
        '../treebanks/UD_Gothic-PROIEL/got_proiel-ud-train.auto',
        '../treebanks/UD_Old_French-SRCMF/fro_srcmf-ud-train.auto',
        '../treebanks/UD_Swedish-Talbanken/sv_talbanken-ud-train.auto',
        '../treebanks/UD_Turkish-BOUN/tr_boun-ud-train.auto',
        '../treebanks/UD_Russian-Taiga/ru_taiga-ud-train.auto',
        '../treebanks/UD_Afrikaans-AfriBooms/af_afribooms-ud-train.auto',
        '../treebanks/UD_Korean-GSD/ko_gsd-ud-train.auto',
        '../treebanks/UD_Galician-TreeGal/gl_treegal-ud-train.auto',
        '../treebanks/UD_Greek-GDT/el_gdt-ud-train.auto',
        '../treebanks/UD_Indonesian-GSD/id_gsd-ud-train.auto',
        '../treebanks/UD_Coptic-Scriptorium/cop_scriptorium-ud-train.auto',
        '../treebanks/UD_Romanian-SiMoNERo/ro_simonero-ud-train.auto',
        '../treebanks/UD_Norwegian-NynorskLIA/no_nynorsklia-ud-train.auto',
        '../treebanks/UD_Portuguese-Bosque/pt_bosque-ud-train.auto',
        '../treebanks/UD_Dutch-Alpino/nl_alpino-ud-train.auto',
        '../treebanks/UD_Estonian-EDT/et_edt-ud-train.auto',
        '../treebanks/UD_English-Atis/en_atis-ud-train.auto',
        '../treebanks/UD_Italian-PoSTWITA/it_postwita-ud-train.auto',
        '../treebanks/UD_Latin-PROIEL/la_proiel-ud-train.auto',
        '../treebanks/UD_Upper_Sorbian-UFAL/hsb_ufal-ud-train.auto',
        '../treebanks/UD_Icelandic-IcePaHC/is_icepahc-ud-train.auto',
        '../treebanks/UD_Russian-SynTagRus/ru_syntagrus-ud-train.auto',
        '../treebanks/UD_Finnish-TDT/fi_tdt-ud-train.auto',
        '../treebanks/UD_Czech-FicTree/cs_fictree-ud-train.auto',
        '../treebanks/UD_French-Rhapsodie/fr_rhapsodie-ud-train.auto',
        '../treebanks/UD_Spanish-GSD/es_gsd-ud-train.auto',
        '../treebanks/UD_Czech-CAC/cs_cac-ud-train.auto',
        '../treebanks/UD_Latin-UDante/la_udante-ud-train.auto',
        '../treebanks/UD_Scottish_Gaelic-ARCOSG/gd_arcosg-ud-train.auto',
        '../treebanks/UD_French-ParTUT/fr_partut-ud-train.auto',
        '../treebanks/UD_Old_East_Slavic-RNC/orv_rnc-ud-train.auto',
        '../treebanks/UD_Italian-VIT/it_vit-ud-train.auto',
        '../treebanks/UD_Spanish-AnCora/es_ancora-ud-train.auto',
        '../treebanks/UD_Turkish-Atis/tr_atis-ud-train.auto',
        '../treebanks/UD_Romanian-RRT/ro_rrt-ud-train.auto',
        '../treebanks/UD_Persian-PerDT/fa_perdt-ud-train.auto',
        '../treebanks/UD_Lithuanian-ALKSNIS/lt_alksnis-ud-train.auto',
        '../treebanks/UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-train.auto',
        '../treebanks/UD_Slovak-SNK/sk_snk-ud-train.auto',
        '../treebanks/UD_Buryat-BDT/bxr_bdt-ud-train.auto',
        '../treebanks/UD_Indonesian-CSUI/id_csui-ud-train.auto',
        '../treebanks/UD_Polish-LFG/pl_lfg-ud-train.auto',
        '../treebanks/UD_German-GSD/de_gsd-ud-train.auto',
        '../treebanks/UD_English-GUM/en_gum-ud-train.auto',
        '../treebanks/UD_Persian-Seraji/fa_seraji-ud-train.auto',
        '../treebanks/UD_Sanskrit-Vedic/sa_vedic-ud-train.auto',
        '../treebanks/UD_Czech-PDT/cs_pdt-ud-train.auto',
        '../treebanks/UD_Hindi-HDTB/hi_hdtb-ud-train.auto',
        '../treebanks/UD_French-ParisStories/fr_parisstories-ud-train.auto',
        '../treebanks/UD_French-GSD/fr_gsd-ud-train.auto',
        '../treebanks/UD_Turkish-Tourism/tr_tourism-ud-train.auto',
        '../treebanks/UD_Turkish-Penn/tr_penn-ud-train.auto',
        '../treebanks/UD_Estonian-EWT/et_ewt-ud-train.auto',
        '../treebanks/UD_Slovenian-SST/sl_sst-ud-train.auto',
        '../treebanks/UD_Hebrew-HTB/he_htb-ud-train.auto',
        '../treebanks/UD_Romanian-Nonstandard/ro_nonstandard-ud-train.auto',
        '../treebanks/UD_North_Sami-Giella/sme_giella-ud-train.auto',
        '../treebanks/UD_Italian-ISDT/it_isdt-ud-train.auto',
        '../treebanks/UD_Latin-Perseus/la_perseus-ud-train.auto',
        '../treebanks/UD_Basque-BDT/eu_bdt-ud-train.auto',
        '../treebanks/UD_Lithuanian-HSE/lt_hse-ud-train.auto',
        '../treebanks/UD_Turkish_German-SAGT/qtd_sagt-ud-train.auto',
        '../treebanks/UD_Russian-GSD/ru_gsd-ud-train.auto',
        '../treebanks/UD_Armenian-ArmTDP/hy_armtdp-ud-train.auto',
        '../treebanks/UD_Ligurian-GLT/lij_glt-ud-train.auto',
        '../treebanks/UD_Livvi-KKPP/olo_kkpp-ud-train.auto',
        '../treebanks/UD_Kurmanji-MG/kmr_mg-ud-train.auto',
        '../treebanks/UD_Kazakh-KTB/kk_ktb-ud-train.auto',
        '../treebanks/UD_Slovenian-SSJ/sl_ssj-ud-train.auto',
        '../treebanks/UD_Ancient_Greek-Perseus/grc_perseus-ud-train.auto',
        '../treebanks/UD_Chinese-GSD/zh_gsd-ud-train.auto',
        '../treebanks/UD_Japanese-GSD/ja_gsd-ud-train.auto',
        '../treebanks/UD_Norwegian-Nynorsk/no_nynorsk-ud-train.auto',
        '../treebanks/UD_Japanese-GSDLUW/ja_gsdluw-ud-train.auto',
        '../treebanks/UD_Serbian-SET/sr_set-ud-train.auto',
        '../treebanks/UD_Swedish-LinES/sv_lines-ud-train.auto',
        '../treebanks/UD_Classical_Chinese-Kyoto/lzh_kyoto-ud-train.auto',
        '../treebanks/UD_Urdu-UDTB/ur_udtb-ud-train.auto',
        '../treebanks/UD_Italian-TWITTIRO/it_twittiro-ud-train.auto',
        '../treebanks/UD_Norwegian-Bokmaal/no_bokmaal-ud-train.auto',
        '../treebanks/UD_Swedish_Sign_Language-SSLC/swl_sslc-ud-train.auto'
    ]
    train_data_items = _load(treebanks_train_data_paths)
    import pickle
    pickle.dump(train_data_items, open('train_data_items.pkl', 'wb'))
    # train_data_items = pickle.load(open('train_data_items.pkl', 'rb'))
    collect_unary_rules(train_data_items,'instantiated_unary_rules_raw.json')
    collect_binary_rules(train_data_items,'instantiated_binary_rules_raw.json')