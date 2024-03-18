from typing import List, Dict, Union, Any, TypeVar
import sys
import argparse
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.append('..')
from data_loader import load_auto_file
from base import Category
from ccg_supertagger.models import BaseSupertaggingModel, LSTMSupertaggingModel
from ccg_supertagger.utils import pre_tokenize_sent


CategoryStr = TypeVar('CategoryStr')
SupertaggerOutput = List[List[CategoryStr]]


DATA_MASK_PADDING = 0


class CCGSupertagger:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        idx2category: Dict[int, str] = None,
        top_k: int = 1,
        beta: float = 1e-5,  # pruning parameter for supertagging
        device: torch.device = torch.device('cuda:0')
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.idx2category = idx2category
        if idx2category is not None:
            self.category2idx = {idx: cat for idx, cat in idx2category.items()}
        self.top_k = top_k
        self.beta = beta
        self.device = device
        self.softmax = nn.Softmax(dim=2)

    def _prepare_batch_data(self, batch: List[List[str]]) -> Dict[str, Any]:
        """
        Input:
            batch - a list of pretokenized sentences (a list of strings)
        Output:
            wrapped and padded batch data to input into the model
        """ 
        data = list()  # a list containing a list of input_ids for each sentence
        mask = list()  # a list containing the attention mask list for each sentence
        word_piece_tracked = list()  # a list containing the list of word_piece_tracked for each sentence

        for pretokenized_sent in batch:
            word_piece_tracked.append(
                [len(item) for item in self.tokenizer(pretokenized_sent, add_special_tokens=False).input_ids]
            )

            inputs = self.tokenizer(
                pretokenized_sent,
                add_special_tokens=False,
                is_split_into_words=True
            )
            data.append(inputs.input_ids)
            mask.append(inputs.attention_mask)

        max_length = max([len(input_ids) for input_ids in data])
        for i in range(len(data)):
            assert len(data[i]) == len(mask[i])
            data[i] = data[i] + [DATA_MASK_PADDING] * \
                (max_length - len(data[i]))  # padding
            mask[i] = mask[i] + [DATA_MASK_PADDING] * \
                (max_length - len(mask[i]))  # padding

        return {
            'input_ids': torch.LongTensor(data),
            'mask': torch.FloatTensor(mask),
            'word_piece_tracked': word_piece_tracked
        }

    def _convert_model_outputs(self, outputs: List[torch.Tensor]) -> List[SupertaggerOutput]:
        """
        Input:
            outputs - a list of tensors, each of shape (the length of one sentence * C)
        Output:
            a list of category lists,
            each of which corresponds to predicted supertags for a sentence
        """
        if self.idx2category is None:
            raise RuntimeError('Please specify idx2category in the supertagger!!!')

        outputs = self._prune(outputs)

        batch_predicted = list()
        for output in outputs:
            predicted = list()
            for i in range(output.shape[0]):
                topk_ps, topk_ids = torch.topk(output[i], self.top_k)
                ids = topk_ids[topk_ps > 0]
                predicted.append(
                    [
                        str(Category.parse(self.idx2category[idx.item()]))
                        for idx in ids
                    ]
                )
            batch_predicted.append(predicted)
        return batch_predicted

    def _prune(self, outputs) -> torch.Tensor:
        # assign all probabilities less than beta times of the best one to 0
        for output in outputs:
            for i in range(output.shape[0]):
                top_p = torch.topk(output[i], 1)[0]
                binarized = (output[i] > self.beta * top_p)
                output[i] = output[i] * binarized

        return outputs

    def _load_model_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_model_outputs_for_batch(self, batch: List[Union[str, List[str]]]) -> List[torch.Tensor]:
        """
        Input:
            batch - a list of sentences (str) or pretokenized sentences (List[str]),
                    better to be pretokenized as the pre_tokenized_sent is not very complete yet
        Output:
            a list of tensors, each of the shape l_sent * C
        """
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()

            for i in range(len(batch)):
                if isinstance(batch[i], str):
                    batch[i] = pre_tokenize_sent(batch[i])

            batch_data = self._prepare_batch_data(batch)
            batch_data['input_ids'] = batch_data['input_ids'].to(self.device)
            batch_data['mask'] = batch_data['mask'].to(self.device)
            outputs = self.model(
                encoded_batch=batch_data['input_ids'],
                mask=batch_data['mask'],
                word_piece_tracked=batch_data['word_piece_tracked']
            )  # B*L*C
            outputs = self.softmax(outputs)

            sents_lengths = [
                len(word_piece_tracked)
                for word_piece_tracked in batch_data['word_piece_tracked']
            ]

            return [
                outputs[i, :sents_lengths[i], :]
                for i in range(len(batch))
            ]

    def get_model_outputs_for_sent(self, sent: Union[str, List[str]]) -> torch.Tensor:
        """
        Input:
            sent - a sentence (str) or a pretokenzied sentence (List[str]),
                   better to be pretokenized as the pre_tokenized_sent is not very complete yet
        Output:
            a tensor of shape (length of this sentence *C)
        """
        return self.get_model_outputs_for_batch([sent])[0]

    def predict_batch(self, batch: List[Union[str, List[str]]]) -> List[SupertaggerOutput]:
        outputs = self.get_model_outputs_for_batch(batch)
        return self._convert_model_outputs(outputs)

    def predict_sent(self, sent: Union[str, List[str]]) -> SupertaggerOutput:
        return self.predict_batch([sent])[0]

    # check the supertagger through re-calculation of the acc
    # can also used for multitagging acc checking
    def sanity_check(
        self,
        pretokenized_sents: List[List[str]],
        golden_supertags: List[List[str]],
        batch_size=10
    ) -> None:
        """
        Input:
            pretokenized_sents - a list of pretokenized sentences (List[str])
            golden_supertags - a list of golden supertag lists,
                               each of which is a list of golden supertag strings
            batch_size - the batch size to be passed into the supertagging model
        """
        correct_cnt = 0
        total_cnt = 0
        n_categories = 0

        for i in range(0, len(pretokenized_sents), batch_size):
            if i % 50 == 0:
                print(f'progress: {i} / {len(pretokenized_sents)}')
            sents = pretokenized_sents[i: i + batch_size]
            supertags = golden_supertags[i: i + batch_size]

            predicted = self.predict_batch(sents)

            total_cnt += sum([len(golden) for golden in supertags])
            for j in range(len(supertags)):
                for k in range(len(supertags[j])):
                    n_categories += len(predicted[j][k])
                    if supertags[j][k] in predicted[j][k]:
                        correct_cnt += 1

        print(
            f'per-word acc of the supertagger = {(correct_cnt / total_cnt) * 100: .3f} (correct if the golden tag is in the top k predicted ones)'
        )
        print(
            f'averaged number of categories per word = {(n_categories / total_cnt): .2f}'
        )


def apply_supertagger(args):

    with open(args.lexical_category2idx_path, 'r', encoding='utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: category for category, idx in category2idx.items()}
    if args.model_name == 'fc':
        model = BaseSupertaggingModel(
            model_dir=args.model_dir,
            n_classes=len(category2idx)
        )
    elif args.model_name == 'lstm': # Not used
        model = LSTMSupertaggingModel(
            model_dir=args.model_dir,
            n_classes=len(category2idx),
            embed_dim=args.embed_dim,
            num_lstm_layers=args.num_lstm_layers
        )
    else:
        raise RuntimeError('Please check the model name!!!')

    supertagger = CCGSupertagger(
        model=model,
        # tokenizer=BertTokenizer.from_pretrained(args.model_dir),
        tokenizer=AutoTokenizer.from_pretrained(args.model_dir),
        idx2category=idx2category,
        top_k=args.top_k,
        beta=args.beta,
        device=torch.device(args.device)
    )
    supertagger._load_model_checkpoint(args.checkpoint_path)

    if args.mode == 'sanity_check':

        def _load(data_paths): # Loading data items from treebanks
            all_data_items = list()
            for path in data_paths:
                try:
                    data_items, _ = load_auto_file(path)
                    all_data_items.extend(data_items)
                except:
                    continue
            return all_data_items

        if args.sanity_check_mode == 'single_treebank':
            # Sanity checking a single treebank
            data_items, _ = load_auto_file(args.sanity_check_data_path)
        elif args.sanity_check_mode == 'all_treebanks_dev':
            # Sanity checking all treebanks in dev data
            data_items = _load(args.treebanks_dev_data_paths)
        elif args.sanity_check_mode == 'all_treebanks_test':
            # Sanity checking all treebanks in test data
            data_items = _load(args.treebanks_test_data_paths)
        elif args.sanity_check_mode == 'ccgbank_dev':
            # Sanity checking dev data from CCGBank
            data_items, _ = load_auto_file(args.ccgbank_dev_data_path)
        elif args.sanity_check_mode == 'ccgbank_test':
            # Sanity checking test data from CCGBank
            data_items, _ = load_auto_file(args.ccgbank_test_data_path)
        else:
            raise RuntimeError('Please check args.sanity_check_mode!!!')

        pretokenized_sents = [
            [token.contents for token in item.tokens]
            for item in data_items
        ]
        golden_supertags = [
            [str(token.tag) for token in item.tokens]
            for item in data_items
        ]

        supertagger.sanity_check(pretokenized_sents, golden_supertags)

    elif args.mode == 'predict':
        # predict supertags of one to many sentences from args.pretokenized_sents_dir
        # and save the results to args.batch_predicted_dir
        with open(args.pretokenized_sents_path, 'r', encoding='utf8') as f:
            pretokenized_sents = json.load(f)
        predicted = supertagger.predict_batch(pretokenized_sents)
        print(predicted)
        with open(args.batch_predicted_path, 'w', encoding='utf8') as f:
            json.dump(predicted, f, indent=2, ensure_ascii=False)

    else:
        raise RuntimeError('Please check the mode of the supertagger!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='apply supertagging')
    parser.add_argument('--sample_data_path', type=str,
                        default='../data/ccg-sample.auto')
    parser.add_argument('--ccgbank_train_data_path', type=str,
                        default='../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--ccgbank_dev_data_path', type=str,
                        default='../data/ccgbank-wsj_00.auto')
    parser.add_argument('--ccgbank_test_data_path', type=str,
                        default='../data/ccgbank-wsj_23.auto')
    parser.add_argument('--sanity_check_data_path', type=str,
                        default='../treebanks/UD_English-Atis/en_atis-ud-dev.auto')
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
                        
    parser.add_argument('--model_dir', type=str,
                        default='../plms/mt5-base')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints_treebanks/checkpoint_treebanks.pt')

    parser.add_argument('--model_name', type=str,
                        default='fc', choices=['fc', 'lstm']) # Not used
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_lstm_layers', type=int, default=1) # Not used
    parser.add_argument('--device', type=str,
                        default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--beta', help='the coefficient used to prune predicted categories',
                        type=float, default=0.0005)

    parser.add_argument('--mode', type=str, default='sanity_check',
                        choices=['predict', 'sanity_check'])
    parser.add_argument('--sanity_check_mode', type=str, default='',
                        choices=['single_treebank', 'all_treebanks_dev', 'all_treebanks_test', 'ccgbank_dev', 'ccgbank_test'])
    parser.add_argument('--pretokenized_sents_path', type=str,
                        default='../data/pretokenized_sents.json')
    parser.add_argument('--batch_predicted_path', type=str,
                        default='./batch_predicted_supertags.json')
    args = parser.parse_args()

    apply_supertagger(args)