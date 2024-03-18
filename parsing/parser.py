from typing import *
import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from ccg_parsing_models import BaseParsingModel
from decoders.decoder import Chart, Decoder
from decoders.ccg_a_star_decoder import AStarChart, CCGAStarDecoder

sys.path.append('..')
from base import Atom, Token, Category, ConstituentNode
from data_loader import DataItem, load_auto_file
from tools import to_auto


class Parser:
    
    def __init__(
        self,
        parsing_model: nn.Module,
        decoder: Decoder
    ):
        self.parsing_model = parsing_model
        self.decoder = decoder

    def batch_parse(
        self,
        pretokenized_sents: List[List[str]]
    ) -> List[Chart]:

        batch_representations = self.parsing_model(pretokenized_sents)
        is_nan = False
        for representation in batch_representations:
            if np.isnan(np.sum(representation.cpu().detach().numpy())):
                is_nan = True
        if not is_nan:
            charts = self.decoder.batch_decode(
                pretokenized_sents, batch_representations
            )
            return charts
        else: # Omit problematic data
            return [
                AStarChart(l_sent=len(sent), idx2tag=None)
                for sent in pretokenized_sents
            ]

    def parse(self, pretokenized_sent: List[str]) -> Chart:
        return self.batch_parse([pretokenized_sent])[0]

    def batch_sanity_check(
        self,
        pretokenized_sents: List[List[str]],
        golden_supertags: List[List[str]],
        print_cell_items: bool = False
    ) -> List[Chart]:
        
        charts = list()
        for i in range(len(pretokenized_sents)):
            charts.append(
                self.decoder.sanity_check(
                    pretokenized_sents[i],
                    golden_supertags[i],
                    print_cell_items
                )
            )
        return charts

    def sanity_check(
        self,
        pretokenized_sent: List[str],
        golden_supertags: List[str],
        print_cell_items: bool = False
    ) -> Chart:
        return self.batch_sanity_check(
            [pretokenized_sent],
            [golden_supertags],
            print_cell_items
        )[0]


def get_batch_data(data_items: List[DataItem]) -> Dict[str, Any]:
    pretokenized_sents = list()
    golden_supertags = list()
    golden_parses = list()
    data_ids = list()
    for data_item in data_items:
        pretokenized_sents.append(
            [
                token.contents
                for token in data_item.tokens
            ]
        )
        golden_supertags.append(
            [
                str(token.tag)
                for token in data_item.tokens
            ]
        )
        golden_parses.append(data_item.tree_root)
        data_ids.append(data_item.id)

    return {
        'pretokenized_sents': pretokenized_sents,
        'golden_supertags': golden_supertags,
        'golden_parses': golden_parses,
        'data_ids': data_ids
    }


def parseval(
    predicted_parses: List[ConstituentNode],
    golden_parses: List[Union[ConstituentNode, None]]
) -> Dict[str, float]:
    """
    Input:
        predicted_parses - a list of ConstituentNodes, each of which is the predicted parse tree of one case
        golden_parses - a list of ConstituentNodes, each of which is the golden parse tree of one case
    Output:
        A dictionary of the form {'unlabelled': {'P': _, 'R': _, 'F1': _}, 'labelled': {'P': _, 'R': _, 'F1': _}},
        storing PARSEVAL results of the predicted parses.
    """

    def _get_start_end(node: Union[Token, ConstituentNode], cnt: List[int]):
        # Calculate the span indices of components in the parse tree
        if isinstance(node, Token):
            node.start_end = (cnt[0], cnt[0] + 1)
            cnt[0] += 1
        elif isinstance(node, ConstituentNode):
            if node.children:
                if len(node.children) == 1:
                    _get_start_end(node.children[0], cnt = cnt)
                    node.start_end = node.children[0].start_end
                elif len(node.children) == 2:
                    _get_start_end(node.children[0], cnt = cnt)
                    _get_start_end(node.children[1], cnt = cnt)
                    node.start_end = (node.children[0].start_end[0], node.children[1].start_end[1])
                else:
                    raise TypeError('Please check the nodes!!!')
        else:
            raise TypeError('Please check the type of the node!!!')

    for idx in range(len(golden_parses)):
        _get_start_end(predicted_parses[idx], cnt=[0])
        _get_start_end(golden_parses[idx], cnt=[0])

    parseval_results_dict = {
        'unlabelled': {'P': None, 'R': None, 'F1': None},
        'labelled': {'P': None, 'R': None, 'F1': None}
    }

    def _linearize(parse: Union[ConstituentNode, Token]):
        # list all constituents in one parse tree, with each constituent represented as (start_pos, end_pos, category_str)
        if isinstance(parse, Token):
            return [(parse.start_end[0], parse.start_end[1], parse.tag)]
        elif isinstance(parse, ConstituentNode):
            constituents = list()
            if parse.children:
                if len(parse.children) == 1:
                    if isinstance(parse.children[0], Token):
                        constituents.extend(_linearize(parse.children[0]))
                    else:
                        constituents.extend(_linearize(parse.children[0]))
                        constituents.append((parse.start_end[0], parse.start_end[1], parse.tag))
                        # print([[c[0], c[1], str(c[2])] for c in constituents])
                elif len(parse.children) == 2:
                    constituents.extend(_linearize(parse.children[0]))
                    constituents.extend(_linearize(parse.children[1]))
                    constituents.append((parse.start_end[0], parse.start_end[1], parse.tag))
                    # print([[c[0], c[1], str(c[2])] for c in constituents])
                else:
                    raise TypeError('Please check the nodes!!!')
            return constituents
        else:
            raise TypeError('Please check the type of the node!!!')

    def _parseval_case(p_parse, g_parse):
        p_constituents = _linearize(p_parse)
        g_constituents = _linearize(g_parse)
        
        matched_spans = []
        matched_l = 0
        for p_c in p_constituents:
            for g_c in g_constituents:
                if p_c[0] == g_c[0] and p_c[1] == g_c[1]:
                    if (p_c[0], p_c[1]) not in matched_spans:
                        matched_spans.append((p_c[0], p_c[1]))
                    if p_c[2] == g_c[2]:
                        matched_l += 1
        
        matched_unl_p = 0
        matched_unl_r = 0
        for p_c in p_constituents:
            if (p_c[0], p_c[1]) in matched_spans:
                matched_unl_p += 1
        for g_c in g_constituents:
            if (g_c[0], g_c[1]) in matched_spans:
                matched_unl_r += 1

        n_p = len(p_constituents)
        n_g = len(g_constituents)

        return matched_unl_p, matched_unl_r, matched_l, n_p, n_g


    matched_unlabelled_p = 0
    matched_unlabelled_r = 0
    matched_labelled = 0
    n_predicted = 0
    n_golden = 0
    for predicted_parse, golden_parse in zip(predicted_parses, golden_parses):
        case_result = _parseval_case(predicted_parse, golden_parse)
        matched_unlabelled_p += case_result[0]
        matched_unlabelled_r += case_result[1]
        matched_labelled += case_result[2]
        n_predicted += case_result[3]
        n_golden += case_result[4]

    parseval_results_dict['unlabelled']['P'] = round(matched_unlabelled_p / n_predicted, 4)
    parseval_results_dict['unlabelled']['R'] = round(matched_unlabelled_r / n_golden, 4)
    parseval_results_dict['unlabelled']['F1'] = round(2 / (1 / parseval_results_dict['unlabelled']['P'] + 1 / parseval_results_dict['unlabelled']['R']), 4)
    parseval_results_dict['labelled']['P'] = round(matched_labelled / n_predicted, 4)
    parseval_results_dict['labelled']['R'] = round(matched_labelled / n_golden, 4)
    parseval_results_dict['labelled']['F1'] = round(2 / (1 / parseval_results_dict['labelled']['P'] + 1 / parseval_results_dict['labelled']['R']), 4)

    return parseval_results_dict


def run(
    batch_data: Dict[str, Any],
    parser: Parser,
    treebank_name: str,
    predicted_auto_saving_path: str,
    eval_results_path: str,
    batch_size: int = 10,
    mode: str = 'predict_batch'
) -> None:
    """
    Input:
        batch_data -  a dictionary storing pretokenized sentences,
                      corresponding golden supertags and data ids
        parser - the parser
        saving_dir - the directory to save the predicted .auto file
        batch_size - the batch size set for supertagging
        mode - the mode specified for batch parsing,
               choices in ['batch_sanity_check', 'predict_batch']
    """
    pretokenized_sents = batch_data['pretokenized_sents']
    golden_supertags = batch_data['golden_supertags']
    golden_parses = batch_data['golden_parses']
    data_ids = batch_data['data_ids']

    accumulated_time = 0
    n_null_parses = 0
    buffer = []
    predicted_parses = list()
    for i in range(0, len(pretokenized_sents), batch_size):
        print(f'======== {i} / {len(pretokenized_sents)} ========')

        t0 = time.time()

        if mode == 'predict_batch':
            charts = parser.batch_parse(
                pretokenized_sents[i: i + batch_size]
            )
        elif mode == 'batch_sanity_check':
            charts = parser.batch_sanity_check(
                pretokenized_sents[i: i + batch_size],
                golden_supertags[i: i + batch_size],
                print_cell_items = False
            )
        else:
            raise RuntimeError('Please check the batch running mode!!!')

        time_cost = time.time() - t0
        print(f'time cost for this batch: {time_cost}s')
        accumulated_time += time_cost

        tmp_data_ids = data_ids[i: i + batch_size]
        for j in range(len(charts)):

            buffer.append(tmp_data_ids[j] + '\n')

            cell_item = None
            if charts[j] is None:
                cell_item = None
            elif charts[j].chart[0][-1].cell_items is None:
                cell_item = None
            elif len(charts[j].chart[0][-1].cell_items) == 0:
                cell_item = None
            else:
                cell_item = charts[j].chart[0][-1].cell_items[-1]
                print(f'Result root for {tmp_data_ids[j]}: {str(cell_item.constituent)}')

            if cell_item:
                buffer.append(to_auto(cell_item.constituent) + '\n')
                predicted_parses.append(cell_item.constituent)
            else:
                buffer.append('(<L S None None None S>)\n')
                predicted_parses.append(ConstituentNode())
                n_null_parses += 1
                print(f'Null parse: {tmp_data_ids[j]}')

    parseval_results_dict = parseval(predicted_parses, golden_parses)

    print(
        f'averaged parsing time of each sentence: {accumulated_time / len(pretokenized_sents)}'
    )

    print(
        f'null parses: {n_null_parses} / {len(pretokenized_sents)} = {n_null_parses / len(pretokenized_sents): .2f}'
    )

    print(parseval_results_dict)

    with open(predicted_auto_saving_path, 'w', encoding='utf8') as f:
        f.writelines(buffer)
    with open(eval_results_path, 'a+', encoding='utf8') as f:
        f.write(
            treebank_name + '\n' \
                + f'null parses: {n_null_parses} / {len(pretokenized_sents)} = {n_null_parses / len(pretokenized_sents): .2f}' + '\n' \
                + str(parseval_results_dict) + '\n'
        )


def apply_parser(args):

    with open(args.lexical_category2idx_path, 'r', encoding = 'utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: cat for cat, idx in category2idx.items()}

    print(
        f'======== AStarParsing_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout} ========'
    )
    decoder = CCGAStarDecoder(
        idx2tag=idx2category,
        possible_roots=args.possible_roots.split('|'),
        top_k=args.top_k_supertags,
        apply_supertagging_pruning=args.apply_supertagging_pruning,
        beta=args.beta,
        timeout=args.decoder_timeout
    )
    
    with open(args.instantiated_unary_rules_path, 'r', encoding = 'utf8') as f:
        instantiated_unary_rules = json.load(f)
    with open(args.instantiated_binary_rules_path, 'r', encoding = 'utf8') as f:
        instantiated_binary_rules = json.load(f)
    decoder._get_instantiated_unary_rules(instantiated_unary_rules, args.unary_rules_n)
    decoder._get_instantiated_binary_rules(instantiated_binary_rules)

    parsing_model = BaseParsingModel(
        model_dir=args.supertagging_model_dir,
        supertagging_n_classes=len(idx2category),
        embed_dim=args.embed_dim,
        checkpoint_path=args.supertagging_model_checkpoint_path,
        device=torch.device(args.device)
    )

    parser = Parser(
        parsing_model = parsing_model,
        decoder = decoder
    )


    def _load(data_paths): # Loading treebank data
        all_data_items_dict = dict()
        for path in data_paths:
            try:
                data_items, _ = load_auto_file(path)
                all_data_items_dict[path.split('/')[-2]] = data_items
            except:
                continue
        return all_data_items_dict


    if args.mode == 'sanity_check':
        data_items, _ = load_auto_file(args.sample_data_path)
        pretokenized_sent = [token.contents for token in data_items[0].tokens]
        golden_supertags = [str(token.tag) for token in data_items[0].tokens]

        chart = parser.sanity_check(pretokenized_sent, golden_supertags, print_cell_items=True)
        
        # print out all successful parses
        for cell_item in chart.chart[0][-1].cell_items:
            print(to_auto(cell_item.constituent))

    elif args.mode == 'predict_sent':
        data_items, _ = load_auto_file(args.sample_data_path)
        pretokenized_sent = [token.contents for token in data_items[0].tokens]
        chart = parser.parse(pretokenized_sent)

        # print out all successful parses
        for cell_item in chart.chart[0][-1].cell_items:
            print(to_auto(cell_item.constituent))

    elif args.mode == 'batch_sanity_check':
        if args.batch_data_mode == '':
            raise RuntimeError('Please check args.batch_data_mode!!!')
        elif args.batch_data_mode.split('_')[0] == 'treebanks':
            if args.batch_data_mode == 'treebanks_dev':
                data_items_dict = _load(args.treebanks_dev_data_paths)
            elif args.batch_data_mode == 'treebanks_test':
                data_items_dict = _load(args.treebanks_test_data_paths)
            else:
                raise RuntimeError('Please check args.batch_data_mode!!!')
            for treebank_name, data_items in data_items_dict.items():
                batch_data = get_batch_data(data_items)
                plm_name = args.supertagging_model_dir.split('/')[-1]
                
                data_mode = args.batch_data_mode.split('_')[1]
                predicted_auto_saving_path = os.path.join(
                    args.predicted_auto_files_dir,
                    f'AStarParsing_TREEBANK{treebank_name}_{data_mode}_PLM{plm_name}_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout}_GOLD.auto'
                )
                eval_results_path = os.path.join(args.predicted_auto_files_dir, 'eval_results_GOLD.txt')
                
                run(
                    batch_data=batch_data,
                    parser=parser,
                    treebank_name=treebank_name,
                    predicted_auto_saving_path=predicted_auto_saving_path,
                    eval_results_path=eval_results_path,
                    batch_size=args.batch_size,
                    mode=args.mode
                )
        elif args.batch_data_mode.split('_')[0] == 'ccgbank':
            if args.batch_data_mode == 'ccgbank_dev':
                data_items, _ = load_auto_file(args.ccgbank_dev_data_path)
            elif args.batch_data_mode == 'ccgbank_test':
                data_items, _ = load_auto_file(args.ccgbank_test_data_path)
            else:
                raise RuntimeError('Please check args.batch_data_mode!!!')
            batch_data = get_batch_data(data_items)
            plm_name = args.supertagging_model_dir.split('/')[-1]
            data_mode = args.batch_data_mode.split('_')[1]
            predicted_auto_saving_path = os.path.join(
                args.predicted_auto_files_dir,
                f'AStarParsing_CCGBANK{data_mode}_PLM{plm_name}_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout}_GOLD.auto'
            )
            eval_results_path = os.path.join(args.predicted_auto_files_dir, 'eval_results_GOLD.txt')
            
            run(
                batch_data=batch_data,
                parser=parser,
                treebank_name=args.batch_data_mode,
                predicted_auto_saving_path=predicted_auto_saving_path,
                eval_results_path=eval_results_path,
                batch_size=args.batch_size,
                mode=args.mode
            )
        else:
            raise RuntimeError('Please check args.batch_data_mode!!!')

    elif args.mode == 'predict_batch':
        if args.batch_data_mode == '':
            raise RuntimeError('Please check args.batch_data_mode!!!')
        elif args.batch_data_mode.split('_')[0] == 'treebanks':
            if args.batch_data_mode == 'treebanks_dev':
                data_items_dict = _load(args.treebanks_dev_data_paths)
            elif args.batch_data_mode == 'treebanks_test':
                data_items_dict = _load(args.treebanks_test_data_paths)
            else:
                raise RuntimeError('Please check args.batch_data_mode!!!')

            for treebank_name, data_items in data_items_dict.items():
                batch_data = get_batch_data(data_items)
                plm_name = args.supertagging_model_dir.split('/')[-1]

                data_mode = args.batch_data_mode.split('_')[1]
                predicted_auto_saving_path = os.path.join(
                    args.predicted_auto_files_dir,
                    f'AStarParsing_TREEBANK{treebank_name}_{data_mode}_PLM{plm_name}_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout}.auto'
                )
                eval_results_path = os.path.join(args.predicted_auto_files_dir, 'eval_results.txt')
                
                run(
                    batch_data=batch_data,
                    parser=parser,
                    treebank_name=treebank_name,
                    predicted_auto_saving_path=predicted_auto_saving_path,
                    eval_results_path=eval_results_path,
                    batch_size=args.batch_size,
                    mode=args.mode
                )
        elif args.batch_data_mode.split('_')[0] == 'ccgbank':
            if args.batch_data_mode == 'ccgbank_dev':
                data_items, _ = load_auto_file(args.ccgbank_dev_data_path)
            elif args.batch_data_mode == 'ccgbank_test':
                data_items, _ = load_auto_file(args.ccgbank_test_data_path)
            else:
                raise RuntimeError('Please check args.batch_data_mode!!!')
            batch_data = get_batch_data(data_items)
            plm_name = args.supertagging_model_dir.split('/')[-1]
            data_mode = args.batch_data_mode.split('_')[1]
            predicted_auto_saving_path = os.path.join(
                args.predicted_auto_files_dir,
                f'AStarParsing_CCGBANK{data_mode}_PLM{plm_name}_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout}.auto'
            )
            eval_results_path = os.path.join(args.predicted_auto_files_dir, 'eval_results.txt')
            
            run(
                batch_data=batch_data,
                parser=parser,
                treebank_name=args.batch_data_mode,
                predicted_auto_saving_path=predicted_auto_saving_path,
                eval_results_path=eval_results_path,
                batch_size=args.batch_size,
                mode=args.mode
            )
        else:
            raise RuntimeError('Please check args.batch_data_mode!!!')
    
    else:
        raise RuntimeError('Please check the args.mode!!!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='apply parsing')

    parser.add_argument('--sample_data_path', type=str, default='../data/sample.auto')
    parser.add_argument('--ccgbank_dev_data_path', type=str, default='../data/ccgbank-wsj_00.auto')
    parser.add_argument('--ccgbank_test_data_path', type=str, default='../data/ccgbank-wsj_23.auto')
    parser.add_argument('--treebanks_dev_data_paths', type=str, nargs='+',
                        default=[
                            '../treebanks/UD_Portuguese-GSD/pt_gsd-ud-dev.auto',
                            '../treebanks/UD_Vietnamese-VTB/vi_vtb-ud-dev.auto',
                            '../treebanks/UD_Telugu-MTG/te_mtg-ud-dev.auto',
                            '../treebanks/UD_Wolof-WTB/wo_wtb-ud-dev.auto',
                            '../treebanks/UD_Catalan-AnCora/ca_ancora-ud-dev.auto',
                            '../treebanks/UD_Turkish-IMST/tr_imst-ud-dev.auto',
                            '../treebanks/UD_Arabic-PADT/ar_padt-ud-dev.auto',
                            '../treebanks/UD_Croatian-SET/hr_set-ud-dev.auto',
                            '../treebanks/UD_Chinese-GSDSimp/zh_gsdsimp-ud-dev.auto',
                            '../treebanks/UD_Marathi-UFAL/mr_ufal-ud-dev.auto',
                            '../treebanks/UD_Turkish-FrameNet/tr_framenet-ud-dev.auto',
                            '../treebanks/UD_Western_Armenian-ArmTDP/hyw_armtdp-ud-dev.auto',
                            '../treebanks/UD_Korean-Kaist/ko_kaist-ud-dev.auto',
                            '../treebanks/UD_Czech-CLTT/cs_cltt-ud-dev.auto',
                            '../treebanks/UD_Welsh-CCG/cy_ccg-ud-dev.auto',
                            '../treebanks/UD_Italian-ParTUT/it_partut-ud-dev.auto',
                            '../treebanks/UD_Finnish-FTB/fi_ftb-ud-dev.auto',
                            '../treebanks/UD_Hungarian-Szeged/hu_szeged-ud-dev.auto',
                            '../treebanks/UD_French-Sequoia/fr_sequoia-ud-dev.auto',
                            '../treebanks/UD_English-ParTUT/en_partut-ud-dev.auto',
                            '../treebanks/UD_Turkish-Kenet/tr_kenet-ud-dev.auto',
                            '../treebanks/UD_Irish-IDT/ga_idt-ud-dev.auto',
                            '../treebanks/UD_Belarusian-HSE/be_hse-ud-dev.auto',
                            '../treebanks/UD_Bulgarian-BTB/bg_btb-ud-dev.auto',
                            '../treebanks/UD_Old_East_Slavic-TOROT/orv_torot-ud-dev.auto',
                            '../treebanks/UD_Dutch-LassySmall/nl_lassysmall-ud-dev.auto',
                            '../treebanks/UD_German-HDT/de_hdt-ud-dev.auto',
                            '../treebanks/UD_Tamil-TTB/ta_ttb-ud-dev.auto',
                            '../treebanks/UD_Maltese-MUDT/mt_mudt-ud-dev.auto',
                            '../treebanks/UD_Icelandic-Modern/is_modern-ud-dev.auto',
                            '../treebanks/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.auto',
                            '../treebanks/UD_Polish-PDB/pl_pdb-ud-dev.auto',
                            '../treebanks/UD_Latin-ITTB/la_ittb-ud-dev.auto',
                            '../treebanks/UD_Ukrainian-IU/uk_iu-ud-dev.auto',
                            '../treebanks/UD_Galician-CTG/gl_ctg-ud-dev.auto',
                            '../treebanks/UD_Uyghur-UDT/ug_udt-ud-dev.auto',
                            '../treebanks/UD_Faroese-FarPaHC/fo_farpahc-ud-dev.auto',
                            '../treebanks/UD_Latin-LLCT/la_llct-ud-dev.auto',
                            '../treebanks/UD_Latvian-LVTB/lv_lvtb-ud-dev.auto',
                            '../treebanks/UD_English-LinES/en_lines-ud-dev.auto',
                            '../treebanks/UD_Danish-DDT/da_ddt-ud-dev.auto',
                            '../treebanks/UD_English-EWT/en_ewt-ud-dev.auto',
                            '../treebanks/UD_Gothic-PROIEL/got_proiel-ud-dev.auto',
                            '../treebanks/UD_Old_French-SRCMF/fro_srcmf-ud-dev.auto',
                            '../treebanks/UD_Swedish-Talbanken/sv_talbanken-ud-dev.auto',
                            '../treebanks/UD_Turkish-BOUN/tr_boun-ud-dev.auto',
                            '../treebanks/UD_Russian-Taiga/ru_taiga-ud-dev.auto',
                            '../treebanks/UD_Afrikaans-AfriBooms/af_afribooms-ud-dev.auto',
                            '../treebanks/UD_Korean-GSD/ko_gsd-ud-dev.auto',
                            '../treebanks/UD_Galician-TreeGal/gl_treegal-ud-dev.auto',
                            '../treebanks/UD_Greek-GDT/el_gdt-ud-dev.auto',
                            '../treebanks/UD_Indonesian-GSD/id_gsd-ud-dev.auto',
                            '../treebanks/UD_Coptic-Scriptorium/cop_scriptorium-ud-dev.auto',
                            '../treebanks/UD_Romanian-SiMoNERo/ro_simonero-ud-dev.auto',
                            '../treebanks/UD_Norwegian-NynorskLIA/no_nynorsklia-ud-dev.auto',
                            '../treebanks/UD_Portuguese-Bosque/pt_bosque-ud-dev.auto',
                            '../treebanks/UD_Dutch-Alpino/nl_alpino-ud-dev.auto',
                            '../treebanks/UD_Estonian-EDT/et_edt-ud-dev.auto',
                            '../treebanks/UD_English-Atis/en_atis-ud-dev.auto',
                            '../treebanks/UD_Italian-PoSTWITA/it_postwita-ud-dev.auto',
                            '../treebanks/UD_Latin-PROIEL/la_proiel-ud-dev.auto',
                            '../treebanks/UD_Upper_Sorbian-UFAL/hsb_ufal-ud-dev.auto',
                            '../treebanks/UD_Icelandic-IcePaHC/is_icepahc-ud-dev.auto',
                            '../treebanks/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.auto',
                            '../treebanks/UD_Finnish-TDT/fi_tdt-ud-dev.auto',
                            '../treebanks/UD_Czech-FicTree/cs_fictree-ud-dev.auto',
                            '../treebanks/UD_French-Rhapsodie/fr_rhapsodie-ud-dev.auto',
                            '../treebanks/UD_Spanish-GSD/es_gsd-ud-dev.auto',
                            '../treebanks/UD_Czech-CAC/cs_cac-ud-dev.auto',
                            '../treebanks/UD_Latin-UDante/la_udante-ud-dev.auto',
                            '../treebanks/UD_Scottish_Gaelic-ARCOSG/gd_arcosg-ud-dev.auto',
                            '../treebanks/UD_French-ParTUT/fr_partut-ud-dev.auto',
                            '../treebanks/UD_Old_East_Slavic-RNC/orv_rnc-ud-dev.auto',
                            '../treebanks/UD_Italian-VIT/it_vit-ud-dev.auto',
                            '../treebanks/UD_Spanish-AnCora/es_ancora-ud-dev.auto',
                            '../treebanks/UD_Turkish-Atis/tr_atis-ud-dev.auto',
                            '../treebanks/UD_Romanian-RRT/ro_rrt-ud-dev.auto',
                            '../treebanks/UD_Persian-PerDT/fa_perdt-ud-dev.auto',
                            '../treebanks/UD_Lithuanian-ALKSNIS/lt_alksnis-ud-dev.auto',
                            '../treebanks/UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-dev.auto',
                            '../treebanks/UD_Slovak-SNK/sk_snk-ud-dev.auto',
                            '../treebanks/UD_Buryat-BDT/bxr_bdt-ud-dev.auto',
                            '../treebanks/UD_Indonesian-CSUI/id_csui-ud-dev.auto',
                            '../treebanks/UD_Polish-LFG/pl_lfg-ud-dev.auto',
                            '../treebanks/UD_German-GSD/de_gsd-ud-dev.auto',
                            '../treebanks/UD_English-GUM/en_gum-ud-dev.auto',
                            '../treebanks/UD_Persian-Seraji/fa_seraji-ud-dev.auto',
                            '../treebanks/UD_Sanskrit-Vedic/sa_vedic-ud-dev.auto',
                            '../treebanks/UD_Czech-PDT/cs_pdt-ud-dev.auto',
                            '../treebanks/UD_Hindi-HDTB/hi_hdtb-ud-dev.auto',
                            '../treebanks/UD_French-ParisStories/fr_parisstories-ud-dev.auto',
                            '../treebanks/UD_French-GSD/fr_gsd-ud-dev.auto',
                            '../treebanks/UD_Turkish-Tourism/tr_tourism-ud-dev.auto',
                            '../treebanks/UD_Turkish-Penn/tr_penn-ud-dev.auto',
                            '../treebanks/UD_Estonian-EWT/et_ewt-ud-dev.auto',
                            '../treebanks/UD_Slovenian-SST/sl_sst-ud-dev.auto',
                            '../treebanks/UD_Hebrew-HTB/he_htb-ud-dev.auto',
                            '../treebanks/UD_Romanian-Nonstandard/ro_nonstandard-ud-dev.auto',
                            '../treebanks/UD_North_Sami-Giella/sme_giella-ud-dev.auto',
                            '../treebanks/UD_Italian-ISDT/it_isdt-ud-dev.auto',
                            '../treebanks/UD_Latin-Perseus/la_perseus-ud-dev.auto',
                            '../treebanks/UD_Basque-BDT/eu_bdt-ud-dev.auto',
                            '../treebanks/UD_Lithuanian-HSE/lt_hse-ud-dev.auto',
                            '../treebanks/UD_Turkish_German-SAGT/qtd_sagt-ud-dev.auto',
                            '../treebanks/UD_Russian-GSD/ru_gsd-ud-dev.auto',
                            '../treebanks/UD_Armenian-ArmTDP/hy_armtdp-ud-dev.auto',
                            '../treebanks/UD_Ligurian-GLT/lij_glt-ud-dev.auto',
                            '../treebanks/UD_Livvi-KKPP/olo_kkpp-ud-dev.auto',
                            '../treebanks/UD_Kurmanji-MG/kmr_mg-ud-dev.auto',
                            '../treebanks/UD_Kazakh-KTB/kk_ktb-ud-dev.auto',
                            '../treebanks/UD_Slovenian-SSJ/sl_ssj-ud-dev.auto',
                            '../treebanks/UD_Ancient_Greek-Perseus/grc_perseus-ud-dev.auto',
                            '../treebanks/UD_Chinese-GSD/zh_gsd-ud-dev.auto',
                            '../treebanks/UD_Japanese-GSD/ja_gsd-ud-dev.auto',
                            '../treebanks/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.auto',
                            '../treebanks/UD_Japanese-GSDLUW/ja_gsdluw-ud-dev.auto',
                            '../treebanks/UD_Serbian-SET/sr_set-ud-dev.auto',
                            '../treebanks/UD_Swedish-LinES/sv_lines-ud-dev.auto',
                            '../treebanks/UD_Classical_Chinese-Kyoto/lzh_kyoto-ud-dev.auto',
                            '../treebanks/UD_Urdu-UDTB/ur_udtb-ud-dev.auto',
                            '../treebanks/UD_Italian-TWITTIRO/it_twittiro-ud-dev.auto',
                            '../treebanks/UD_Norwegian-Bokmaal/no_bokmaal-ud-dev.auto',
                            '../treebanks/UD_Swedish_Sign_Language-SSLC/swl_sslc-ud-dev.auto'
                        ], help='The list of paths to dev data of treebanks')
    parser.add_argument('--treebanks_test_data_paths', type=str, nargs='+',
                        default=[
                            '../treebanks/UD_Portuguese-GSD/pt_gsd-ud-test.auto',
                            '../treebanks/UD_Vietnamese-VTB/vi_vtb-ud-test.auto',
                            '../treebanks/UD_Telugu-MTG/te_mtg-ud-test.auto',
                            '../treebanks/UD_Wolof-WTB/wo_wtb-ud-test.auto',
                            '../treebanks/UD_Catalan-AnCora/ca_ancora-ud-test.auto',
                            '../treebanks/UD_Turkish-IMST/tr_imst-ud-test.auto',
                            '../treebanks/UD_Arabic-PADT/ar_padt-ud-test.auto',
                            '../treebanks/UD_Croatian-SET/hr_set-ud-test.auto',
                            '../treebanks/UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.auto',
                            '../treebanks/UD_Marathi-UFAL/mr_ufal-ud-test.auto',
                            '../treebanks/UD_Turkish-FrameNet/tr_framenet-ud-test.auto',
                            '../treebanks/UD_Western_Armenian-ArmTDP/hyw_armtdp-ud-test.auto',
                            '../treebanks/UD_Korean-Kaist/ko_kaist-ud-test.auto',
                            '../treebanks/UD_Czech-CLTT/cs_cltt-ud-test.auto',
                            '../treebanks/UD_Welsh-CCG/cy_ccg-ud-test.auto',
                            '../treebanks/UD_Italian-ParTUT/it_partut-ud-test.auto',
                            '../treebanks/UD_Finnish-FTB/fi_ftb-ud-test.auto',
                            '../treebanks/UD_Hungarian-Szeged/hu_szeged-ud-test.auto',
                            '../treebanks/UD_French-Sequoia/fr_sequoia-ud-test.auto',
                            '../treebanks/UD_English-ParTUT/en_partut-ud-test.auto',
                            '../treebanks/UD_Turkish-Kenet/tr_kenet-ud-test.auto',
                            '../treebanks/UD_Irish-IDT/ga_idt-ud-test.auto',
                            '../treebanks/UD_Belarusian-HSE/be_hse-ud-test.auto',
                            '../treebanks/UD_Bulgarian-BTB/bg_btb-ud-test.auto',
                            '../treebanks/UD_Old_East_Slavic-TOROT/orv_torot-ud-test.auto',
                            '../treebanks/UD_Dutch-LassySmall/nl_lassysmall-ud-test.auto',
                            '../treebanks/UD_German-HDT/de_hdt-ud-test.auto',
                            '../treebanks/UD_Tamil-TTB/ta_ttb-ud-test.auto',
                            '../treebanks/UD_Maltese-MUDT/mt_mudt-ud-test.auto',
                            '../treebanks/UD_Icelandic-Modern/is_modern-ud-test.auto',
                            '../treebanks/UD_Ancient_Greek-PROIEL/grc_proiel-ud-test.auto',
                            '../treebanks/UD_Polish-PDB/pl_pdb-ud-test.auto',
                            '../treebanks/UD_Latin-ITTB/la_ittb-ud-test.auto',
                            '../treebanks/UD_Ukrainian-IU/uk_iu-ud-test.auto',
                            '../treebanks/UD_Galician-CTG/gl_ctg-ud-test.auto',
                            '../treebanks/UD_Uyghur-UDT/ug_udt-ud-test.auto',
                            '../treebanks/UD_Faroese-FarPaHC/fo_farpahc-ud-test.auto',
                            '../treebanks/UD_Latin-LLCT/la_llct-ud-test.auto',
                            '../treebanks/UD_Latvian-LVTB/lv_lvtb-ud-test.auto',
                            '../treebanks/UD_English-LinES/en_lines-ud-test.auto',
                            '../treebanks/UD_Danish-DDT/da_ddt-ud-test.auto',
                            '../treebanks/UD_English-EWT/en_ewt-ud-test.auto',
                            '../treebanks/UD_Gothic-PROIEL/got_proiel-ud-test.auto',
                            '../treebanks/UD_Old_French-SRCMF/fro_srcmf-ud-test.auto',
                            '../treebanks/UD_Swedish-Talbanken/sv_talbanken-ud-test.auto',
                            '../treebanks/UD_Turkish-BOUN/tr_boun-ud-test.auto',
                            '../treebanks/UD_Russian-Taiga/ru_taiga-ud-test.auto',
                            '../treebanks/UD_Afrikaans-AfriBooms/af_afribooms-ud-test.auto',
                            '../treebanks/UD_Korean-GSD/ko_gsd-ud-test.auto',
                            '../treebanks/UD_Galician-TreeGal/gl_treegal-ud-test.auto',
                            '../treebanks/UD_Greek-GDT/el_gdt-ud-test.auto',
                            '../treebanks/UD_Indonesian-GSD/id_gsd-ud-test.auto',
                            '../treebanks/UD_Coptic-Scriptorium/cop_scriptorium-ud-test.auto',
                            '../treebanks/UD_Romanian-SiMoNERo/ro_simonero-ud-test.auto',
                            '../treebanks/UD_Norwegian-NynorskLIA/no_nynorsklia-ud-test.auto',
                            '../treebanks/UD_Portuguese-Bosque/pt_bosque-ud-test.auto',
                            '../treebanks/UD_Dutch-Alpino/nl_alpino-ud-test.auto',
                            '../treebanks/UD_Estonian-EDT/et_edt-ud-test.auto',
                            '../treebanks/UD_English-Atis/en_atis-ud-test.auto',
                            '../treebanks/UD_Italian-PoSTWITA/it_postwita-ud-test.auto',
                            '../treebanks/UD_Latin-PROIEL/la_proiel-ud-test.auto',
                            '../treebanks/UD_Upper_Sorbian-UFAL/hsb_ufal-ud-test.auto',
                            '../treebanks/UD_Icelandic-IcePaHC/is_icepahc-ud-test.auto',
                            '../treebanks/UD_Russian-SynTagRus/ru_syntagrus-ud-test.auto',
                            '../treebanks/UD_Finnish-TDT/fi_tdt-ud-test.auto',
                            '../treebanks/UD_Czech-FicTree/cs_fictree-ud-test.auto',
                            '../treebanks/UD_French-Rhapsodie/fr_rhapsodie-ud-test.auto',
                            '../treebanks/UD_Spanish-GSD/es_gsd-ud-test.auto',
                            '../treebanks/UD_Czech-CAC/cs_cac-ud-test.auto',
                            '../treebanks/UD_Latin-UDante/la_udante-ud-test.auto',
                            '../treebanks/UD_Scottish_Gaelic-ARCOSG/gd_arcosg-ud-test.auto',
                            '../treebanks/UD_French-ParTUT/fr_partut-ud-test.auto',
                            '../treebanks/UD_Old_East_Slavic-RNC/orv_rnc-ud-test.auto',
                            '../treebanks/UD_Italian-VIT/it_vit-ud-test.auto',
                            '../treebanks/UD_Spanish-AnCora/es_ancora-ud-test.auto',
                            '../treebanks/UD_Turkish-Atis/tr_atis-ud-test.auto',
                            '../treebanks/UD_Romanian-RRT/ro_rrt-ud-test.auto',
                            '../treebanks/UD_Persian-PerDT/fa_perdt-ud-test.auto',
                            '../treebanks/UD_Lithuanian-ALKSNIS/lt_alksnis-ud-test.auto',
                            '../treebanks/UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-test.auto',
                            '../treebanks/UD_Slovak-SNK/sk_snk-ud-test.auto',
                            '../treebanks/UD_Buryat-BDT/bxr_bdt-ud-test.auto',
                            '../treebanks/UD_Indonesian-CSUI/id_csui-ud-test.auto',
                            '../treebanks/UD_Polish-LFG/pl_lfg-ud-test.auto',
                            '../treebanks/UD_German-GSD/de_gsd-ud-test.auto',
                            '../treebanks/UD_English-GUM/en_gum-ud-test.auto',
                            '../treebanks/UD_Persian-Seraji/fa_seraji-ud-test.auto',
                            '../treebanks/UD_Sanskrit-Vedic/sa_vedic-ud-test.auto',
                            '../treebanks/UD_Czech-PDT/cs_pdt-ud-test.auto',
                            '../treebanks/UD_Hindi-HDTB/hi_hdtb-ud-test.auto',
                            '../treebanks/UD_French-ParisStories/fr_parisstories-ud-test.auto',
                            '../treebanks/UD_French-GSD/fr_gsd-ud-test.auto',
                            '../treebanks/UD_Turkish-Tourism/tr_tourism-ud-test.auto',
                            '../treebanks/UD_Turkish-Penn/tr_penn-ud-test.auto',
                            '../treebanks/UD_Estonian-EWT/et_ewt-ud-test.auto',
                            '../treebanks/UD_Slovenian-SST/sl_sst-ud-test.auto',
                            '../treebanks/UD_Hebrew-HTB/he_htb-ud-test.auto',
                            '../treebanks/UD_Romanian-Nonstandard/ro_nonstandard-ud-test.auto',
                            '../treebanks/UD_North_Sami-Giella/sme_giella-ud-test.auto',
                            '../treebanks/UD_Italian-ISDT/it_isdt-ud-test.auto',
                            '../treebanks/UD_Latin-Perseus/la_perseus-ud-test.auto',
                            '../treebanks/UD_Basque-BDT/eu_bdt-ud-test.auto',
                            '../treebanks/UD_Lithuanian-HSE/lt_hse-ud-test.auto',
                            '../treebanks/UD_Turkish_German-SAGT/qtd_sagt-ud-test.auto',
                            '../treebanks/UD_Russian-GSD/ru_gsd-ud-test.auto',
                            '../treebanks/UD_Armenian-ArmTDP/hy_armtdp-ud-test.auto',
                            '../treebanks/UD_Ligurian-GLT/lij_glt-ud-test.auto',
                            '../treebanks/UD_Livvi-KKPP/olo_kkpp-ud-test.auto',
                            '../treebanks/UD_Kurmanji-MG/kmr_mg-ud-test.auto',
                            '../treebanks/UD_Kazakh-KTB/kk_ktb-ud-test.auto',
                            '../treebanks/UD_Slovenian-SSJ/sl_ssj-ud-test.auto',
                            '../treebanks/UD_Ancient_Greek-Perseus/grc_perseus-ud-test.auto',
                            '../treebanks/UD_Chinese-GSD/zh_gsd-ud-test.auto',
                            '../treebanks/UD_Japanese-GSD/ja_gsd-ud-test.auto',
                            '../treebanks/UD_Norwegian-Nynorsk/no_nynorsk-ud-test.auto',
                            '../treebanks/UD_Japanese-GSDLUW/ja_gsdluw-ud-test.auto',
                            '../treebanks/UD_Serbian-SET/sr_set-ud-test.auto',
                            '../treebanks/UD_Swedish-LinES/sv_lines-ud-test.auto',
                            '../treebanks/UD_Classical_Chinese-Kyoto/lzh_kyoto-ud-test.auto',
                            '../treebanks/UD_Urdu-UDTB/ur_udtb-ud-test.auto',
                            '../treebanks/UD_Italian-TWITTIRO/it_twittiro-ud-test.auto',
                            '../treebanks/UD_Norwegian-Bokmaal/no_bokmaal-ud-test.auto',
                            '../treebanks/UD_Swedish_Sign_Language-SSLC/swl_sslc-ud-test.auto'
                        ], help='The list of paths to test data of treebanks')

    parser.add_argument('--lexical_category2idx_path', type=str,
                        default='../data/lexical_category2idx_cutoff_treebanks.json')
    parser.add_argument('--instantiated_unary_rules_path', type=str,
                        default='../data/instantiated_unary_rules.json')
    parser.add_argument('--unary_rules_n', type=int, default=20, help='Taking the n-top unary rules')
    parser.add_argument('--instantiated_binary_rules_path', type=str,
                        default='../data/instantiated_binary_rules.json')
                        
    parser.add_argument('--supertagging_model_dir', type=str,
                        default='../plms/mt5-base')
    parser.add_argument('--supertagging_model_checkpoint_path',
                        type=str, default='../ccg_supertagger/checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt')
    parser.add_argument('--predicted_auto_files_dir',
                        type=str, default='./evaluation')

    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_lstm_layers', type=int, default=1) # Not used
    parser.add_argument('--apply_supertagging_pruning', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--top_k_supertags', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--decoder_timeout', help='time limit for decoding one sentence',
                        type=float, default=16.0)
    parser.add_argument('--possible_roots', help='possible categories at the roots of parses',
                        type=str, default='S|NP|S/NP|S\\NP')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--mode', type=str, default='',
                        choices=['sanity_check', 'predict_sent', 'batch_sanity_check', 'predict_batch'])
    parser.add_argument('--batch_data_mode', type=str, default='',
                        choices=['treebanks_dev', 'treebanks_test', 'ccgbank_dev', 'ccgbank_test'], help='Whether to use data of treebanks or CCGBank')

    args = parser.parse_args()

    apply_parser(args)