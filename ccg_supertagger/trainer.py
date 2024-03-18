from typing import Callable, Dict, List, Tuple
import sys
import os
import argparse
import random
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from models import BaseSupertaggingModel, LSTMSupertaggingModel, LSTMCRFSupertaggingModel
from utils import prepare_data, calculate_treebanks_statistics, results_printer

sys.path.append('..')
from data_loader import load_auto_file


# to set the random seeds
def _setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


_setup_seed(0)


# to set the random seed of the dataloader
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class CCGSupertaggingDataset(Dataset):
    def __init__(self, ids, data, mask, word_piece_tracked, target):
        self.ids = ids
        self.data = data
        self.mask = mask
        self.word_piece_tracked = word_piece_tracked
        self.target = target

    def __getitem__(self, idx):
        return self.ids[idx], self.data[idx], self.mask[idx], self.word_piece_tracked[idx], self.target[idx]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    batch = list(zip(*batch))

    ids = batch[0]
    data = torch.stack(batch[1])
    mask = torch.stack(batch[2])
    word_piece_tracked = batch[3]
    target = torch.stack(batch[4])

    del batch
    return ids, data, mask, word_piece_tracked, target


class CCGSupertaggingTrainer:
    def __init__(
        self,
        n_epochs: int,
        device: torch.device,
        model: nn.Module,
        batch_size: int,
        checkpoints_dir: str,
        checkpoint_name: str,
        train_dataset: Dataset = None,
        train_treebanks_statistics: List[Tuple[str, int, int, int]] = None,
        dev_dataset: Dataset = None,
        dev_datasets_dict: Dict[str, Dataset] = None,
        dev_treebanks_statistics: List[Tuple[str, int, int, int]] = None,
        test_dataset: Dataset = None,
        test_datasets_dict: Dict[str, Dataset] = None,
        test_treebanks_statistics: List[Tuple[str, int, int, int]] = None,
        optimizer: torch.optim = AdamW,
        lr=0.00001,
    ):
        self.n_epochs = n_epochs
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.checkpoints_dir = checkpoints_dir
        self.checkpoint_name = checkpoint_name
        self.train_dataset = train_dataset
        self.train_treebanks_statistics = train_treebanks_statistics
        self.dev_dataset = dev_dataset
        self.dev_datasets_dict = dev_datasets_dict
        self.dev_treebanks_statistics = dev_treebanks_statistics
        self.test_dataset = test_dataset
        self.test_datasets_dict = test_datasets_dict
        self.test_treebanks_statistics = test_treebanks_statistics
        self.optimizer = optimizer
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=2)

    def train(self, checkpoint_epoch: int = 0, print_every: int = 50):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
            num_workers=0,
            generator=g
        )

        self.model.to(self.device)

        if isinstance(self.optimizer, Callable):
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = self.optimizer(params, lr=self.lr)

        best_epoch = -1
        best_dev_acc = -1

        for epoch in range(checkpoint_epoch + 1, self.n_epochs + 1):
            self.model.train()
            i = 0
            for ids, data, mask, word_piece_tracked, target in train_dataloader:
                i += 1

                data = data.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                if self.model.__class__.__name__ == 'LSTMCRFSupertaggingModel': # Not used
                    outputs = self.model(data, target, mask, word_piece_tracked)
                    loss = outputs
                else:
                    outputs = self.model(data, mask, word_piece_tracked)

                    outputs_ = outputs.view(-1, outputs.size(-1))
                    if np.isnan(np.sum(outputs_.cpu().detach().numpy())): # Print ids of problematic data
                        print(ids)
                        continue
                    target_ = target.view(-1)
                    loss = self.criterion(outputs_, target_)

                if i % print_every == 0:
                    print(
                        f'[epoch {epoch}/{self.n_epochs}] averaged training loss of batch {i}/{len(train_dataloader)} = {loss.item()}'
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(
                f'\n======== [epoch {epoch}/{self.n_epochs}] saving the checkpoint ========\n'
            )
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },
                f=os.path.join(
                    self.checkpoints_dir,
                    self.checkpoint_name + f'_epoch_{epoch}.pt'
                )
            )

            with torch.no_grad():
                print(f'\n======== [epoch {epoch}/{self.n_epochs}] train data evaluation ========\n')
                _ = self.test(self.train_dataset, mode='train_eval')
                print(f'\n======== [epoch {epoch}/{self.n_epochs}] dev data evaluation ========\n')
                _, dev_acc = self.test(self.dev_dataset, mode='dev_eval')

                if dev_acc > best_dev_acc:
                    best_epoch = epoch
                    best_dev_acc = dev_acc

        print('\n#Params: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print(f'\nBest epoch = {best_epoch} with dev_eval acc = {best_dev_acc}\n')

    def test(self, dataset: Dataset, mode: str):
        """
        Choose one of the three as the mode:
        ['train_eval', 'dev_eval', 'test_eval']
        """
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
            num_workers=0,
            generator=g
        )

        self.model.to(self.device)
        self.model.eval()

        loss_sum = 0.
        correct_cnt = 0
        total_cnt = 0

        i = 0
        for ids, data, mask, word_piece_tracked, target in dataloader:
            i += 1
            if i % 50 == 0:
                print(f'{mode} progress: {i}/{len(dataloader)}')

            data = data.to(self.device)
            mask = mask.to(self.device)
            target = target.to(self.device)

            if self.model.__class__.__name__ == 'LSTMCRFSupertaggingModel': # Not used
                outputs = self.model(data, target, mask, word_piece_tracked)
                loss = outputs

                predicted_tags = self.model.predict(data, mask, word_piece_tracked)

                predicted = torch.empty_like(target).fill_(-1)
                for j in range(predicted.shape[0]):
                    predicted[j, 0:len(predicted_tags[j])] = torch.tensor(predicted_tags[j])

                correct_cnt += (predicted == target).sum()
            else:
                outputs = self.model(data, mask, word_piece_tracked)

                outputs_ = outputs.view(-1, outputs.size(-1))
                if np.isnan(np.sum(outputs_.cpu().detach().numpy())): # Print ids of problematic data
                    print(ids)
                    continue
                target_ = target.view(-1)
                loss = self.criterion(outputs_, target_)

                outputs = self.softmax(outputs)
                correct_cnt += (torch.argmax(outputs, dim=2) == target).sum()

            loss_sum += loss.item()

            total_cnt += sum(
                [
                    len(word_pieces) for word_pieces in word_piece_tracked
                ]
            )

        loss_sum /= len(dataloader)
        acc = (correct_cnt / total_cnt) * 100
        print(f'averaged {mode} loss = {loss_sum}')
        print(f'{mode} acc = {acc: .3f}')

        return (loss_sum, acc)

    def load_checkpoint_and_train(self, checkpoint_epoch: int):
        """
        Input (checkpoint_epoch): set the epoch from which to restart training
        """
        checkpoint = torch.load(
            os.path.join(
                self.checkpoints_dir,
                self.checkpoint_name + f'_epoch_{checkpoint_epoch}.pt'
            ),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.optimizer(params, lr=self.lr)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']

        self.train(checkpoint_epoch=epoch)

    def load_checkpoint_and_test(
        self, checkpoint_epoch: int, mode: str, data_mode: str
    ):
        """
        Choose one of the three modes as the mode: ['train_eval', 'dev_eval', 'test_eval']
        Choose one of the two data modes: ['treebanks', 'ccgbank']
        """
        checkpoint = torch.load(
            os.path.join(
                self.checkpoints_dir,
                self.checkpoint_name + f'_epoch_{checkpoint_epoch}.pt'
            ),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        with torch.no_grad():
            if mode == 'train_eval':
                self.test(self.train_dataset, mode='train_eval')
            elif mode == 'dev_eval':
                if data_mode == 'treebanks':
                    # When loading checkpoints and testing dev data, only report results on each single treebank
                    evaluation_results = dict()
                    for key in self.dev_datasets_dict:
                        print(f'================= testing on {key} =================\n')
                        evaluation_results[key] = self.test(self.dev_datasets_dict[key], mode='dev_eval')
                    results_printer(
                        train_treebanks_statistics=self.train_treebanks_statistics,
                        test_treebanks_statistics=self.dev_treebanks_statistics,
                        evaluation_results=evaluation_results
                    )
                elif data_mode == 'ccgbank':
                    self.test(self.dev_dataset, mode='dev_eval')
                else:
                    raise RuntimeError('Please check data_mode of testing!!!')
            elif mode == 'test_eval':
                if data_mode == 'treebanks':
                    # When testing test data, only report results on each single treebank
                    evaluation_results = dict()
                    for key in self.test_datasets_dict:
                        print(f'================= testing on {key} =================\n')
                        evaluation_results[key] = self.test(self.test_datasets_dict[key], mode='test_eval')
                    results_printer(
                        train_treebanks_statistics=self.train_treebanks_statistics,
                        test_treebanks_statistics=self.test_treebanks_statistics,
                        evaluation_results=evaluation_results
                    )
                elif data_mode == 'ccgbank':
                    self.test(self.test_dataset, mode='test_eval')
                else:
                    raise RuntimeError('Please check data_mode of testing!!!')
            else:
                raise ValueError('the mode should be one of train_eval, dev_eval and test_eval')


def main(args):
    
    with open('problematic_ids.txt', 'r', encoding='utf8') as f:
        problematic_ids = [line.strip() for line in f.readlines()]


    def _load(data_paths): # Loading while deleting problematic training cases
        all_data_items = list()
        for path in data_paths:
            try:
                data_items, _ = load_auto_file(path, problematic_ids)
                all_data_items.extend(data_items)
            except:
                continue
        return all_data_items


    print('================= loading data items =================\n')
    if args.load_mode == 'first':
        if args.data_mode == 'treebanks':
            # For treebanks
            train_data_items = _load(args.treebanks_train_data_paths)
            dev_data_items = _load(args.treebanks_dev_data_paths)
            test_data_items = _load(args.treebanks_test_data_paths)
        elif args.data_mode == 'ccgbank':
            # For CCGBank
            train_data_items = _load([args.ccgbank_train_data_path])
            dev_data_items = _load([args.ccgbank_dev_data_path])
            test_data_items = _load([args.ccgbank_test_data_path])
        else:
            raise RuntimeError('Please check args.data_mode!!!')
    elif args.load_mode == 'reuse':
        pass # No saved .pkl files for data items
    else:
        raise RuntimeError('Please check args.load_mode!!!')


    if args.data_mode == 'treebanks':
        print('================= loading treebanks\' statistics =================\n')
        # We will sort supertagging evaluation results according to treebanks' statistics
        if args.load_mode == 'first':
            train_treebanks_statistics = calculate_treebanks_statistics(args.treebanks_train_data_paths)
            dev_treebanks_statistics = calculate_treebanks_statistics(args.treebanks_dev_data_paths)
            test_treebanks_statistics = calculate_treebanks_statistics(args.treebanks_test_data_paths)
            # Saving treebanks' statistics
            pickle.dump(train_treebanks_statistics, open('train_treebanks_statistics.pkl', 'wb'))
            pickle.dump(dev_treebanks_statistics, open('dev_treebanks_statistics.pkl', 'wb'))
            pickle.dump(test_treebanks_statistics, open('test_treebanks_statistics.pkl', 'wb'))
        elif args.load_mode == 'reuse':
            # For reusing treebanks' statistics
            train_treebanks_statistics = pickle.load(open('train_treebanks_statistics.pkl', 'rb'))
            dev_treebanks_statistics = pickle.load(open('dev_treebanks_statistics.pkl', 'rb'))
            test_treebanks_statistics = pickle.load(open('test_treebanks_statistics.pkl', 'rb'))
        else:
            raise RuntimeError('Please check args.load_mode!!!')
    elif args.data_mode == 'ccgbank':
        pass # No statistics for CCGBank
    else:
        raise RuntimeError('Please check args.data_mode!!!')


    if args.data_mode == 'treebanks':
        print('================= loading data items dict =================\n')
        # Data items dict is for logging supertagging evaluation results for each treebank
        if args.load_mode == 'first':
            dev_data_items_dict = dict()
            for path in args.treebanks_dev_data_paths:
                try:
                    dev_data_items_, _ = load_auto_file(path)
                except:
                    print('No dev data for:', path)
                    continue
                dev_data_items_dict[path.split('/')[-2]] = dev_data_items_

            test_data_items_dict = dict()
            for path in args.treebanks_test_data_paths:
                try:
                    test_data_items_, _ = load_auto_file(path)
                except:
                    print('No test data for:', path)
                    continue
                test_data_items_dict[path.split('/')[-2]] = test_data_items_
        elif args.load_mode == 'reuse':
            pass # No saved .pkl files for data items dicts
        else:
            raise RuntimeError('Please check args.load_mode!!!')
    elif args.data_mode == 'ccgbank':
        pass # No data_items_dict for CCGBank
    else:
        raise RuntimeError('Please check args.data_mode!!!')


    print('================= loading category2idx dict =================\n')
    with open(args.lexical_category2idx_path, 'r', encoding='utf8') as f:
        category2idx = json.load(f)


    print('================= preparing data =================\n')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    if args.load_mode == 'first':
        train_data = prepare_data(train_data_items, tokenizer, category2idx)
        dev_data = prepare_data(dev_data_items, tokenizer, category2idx)
        if args.data_mode == 'treebanks':
            dev_data_dict = dict()
            for key in dev_data_items_dict.keys():
                dev_data_dict[key] = prepare_data(dev_data_items_dict[key], tokenizer, category2idx)
            test_data_dict = dict()
            for key in test_data_items_dict.keys():
                test_data_dict[key] = prepare_data(test_data_items_dict[key], tokenizer, category2idx)
            pickle.dump(train_data, open('train_data_treebanks.pkl', 'wb'))
            pickle.dump(dev_data, open('dev_data_treebanks.pkl', 'wb'))
            pickle.dump(dev_data_dict, open('dev_data_dict.pkl', 'wb'))
            pickle.dump(test_data_dict, open('test_data_dict.pkl', 'wb'))
        elif args.data_mode == 'ccgbank':
            test_data = prepare_data(test_data_items, tokenizer, category2idx)
            pickle.dump(train_data, open('train_data_ccgbank.pkl', 'wb'))
            pickle.dump(dev_data, open('dev_data_ccgbank.pkl', 'wb'))
            pickle.dump(test_data, open('test_data_ccgbank.pkl', 'wb'))
        else:
            raise RuntimeError('Please check args.data_mode!!!')
    elif args.load_mode == 'reuse':
        if args.data_mode == 'treebanks':
            train_data = pickle.load(open('train_data_treebanks.pkl', 'rb'))
            dev_data = pickle.load(open('dev_data_treebanks.pkl', 'rb'))
            dev_data_dict = pickle.load(open('dev_data_dict.pkl', 'rb'))
            test_data_dict = pickle.load(open('test_data_dict.pkl', 'rb'))
        elif args.data_mode == 'ccgbank':
            train_data = pickle.load(open('train_data_ccgbank.pkl', 'rb'))
            dev_data = pickle.load(open('dev_data_ccgbank.pkl', 'rb'))
            test_data = pickle.load(open('test_data_ccgbank.pkl', 'rb'))
        else:
            raise RuntimeError('Please check args.data_mode!!!')
    else:
        raise RuntimeError('Please check args.load_mode!!!')


    print('================= preparing datasets =================\n')
    train_dataset = CCGSupertaggingDataset(
        ids=train_data['ids'],
        data=train_data['input_ids'],
        mask=train_data['mask'],
        word_piece_tracked=train_data['word_piece_tracked'],
        target=train_data['target']
    )
    dev_dataset = CCGSupertaggingDataset(
        ids=dev_data['ids'],
        data=dev_data['input_ids'],
        mask=dev_data['mask'],
        word_piece_tracked=dev_data['word_piece_tracked'],
        target=dev_data['target']
    )

    if args.data_mode == 'treebanks':
        dev_datasets_dict = dict()
        for key in dev_data_dict.keys():
            dev_datasets_dict[key] = CCGSupertaggingDataset(
                ids=dev_data_dict[key]['ids'],
                data=dev_data_dict[key]['input_ids'],
                mask=dev_data_dict[key]['mask'],
                word_piece_tracked=dev_data_dict[key]['word_piece_tracked'],
                target=dev_data_dict[key]['target']
            )
        test_datasets_dict = dict()
        for key in test_data_dict.keys():
            test_datasets_dict[key] = CCGSupertaggingDataset(
                ids=test_data_dict[key]['ids'],
                data=test_data_dict[key]['input_ids'],
                mask=test_data_dict[key]['mask'],
                word_piece_tracked=test_data_dict[key]['word_piece_tracked'],
                target=test_data_dict[key]['target']
            )
    elif args.data_mode == 'ccgbank':
        test_dataset = CCGSupertaggingDataset(
            ids=test_data['ids'],
            data=test_data['input_ids'],
            mask=test_data['mask'],
            word_piece_tracked=test_data['word_piece_tracked'],
            target=test_data['target']
        )
    else:
        raise RuntimeError('Please check args.data_mode!!!')


    model = BaseSupertaggingModel
    if args.model_name == 'lstm': # Not used
        model = LSTMSupertaggingModel
    elif args.model_name == 'lstm-crf': # Not used
        model = LSTMCRFSupertaggingModel


    if args.data_mode == 'treebanks':
        trainer = CCGSupertaggingTrainer(
            n_epochs=args.n_epochs,
            device=torch.device(args.device),
            model=model(
                model_dir=args.model_dir,
                n_classes=len(category2idx),
                embed_dim=args.embed_dim,
                num_lstm_layers=args.num_lstm_layers,
                dropout_p=args.dropout_p
            ),
            batch_size=args.batch_size,
            checkpoints_dir=args.checkpoints_dir,
            checkpoint_name='_'.join(
                [
                    args.model_name,
                    args.model_dir.split('/')[-1],
                    'drop' + str(args.dropout_p)
                ]
            ),
            train_dataset=train_dataset,
            train_treebanks_statistics=train_treebanks_statistics,
            dev_dataset=dev_dataset,
            dev_datasets_dict=dev_datasets_dict,
            dev_treebanks_statistics=dev_treebanks_statistics,
            test_datasets_dict=test_datasets_dict,
            test_treebanks_statistics=test_treebanks_statistics,
            lr=args.lr
        )
    elif args.data_mode == 'ccgbank':
        trainer = CCGSupertaggingTrainer(
            n_epochs=args.n_epochs,
            device=torch.device(args.device),
            model=model(
                model_dir=args.model_dir,
                n_classes=len(category2idx),
                embed_dim=args.embed_dim,
                num_lstm_layers=args.num_lstm_layers,
                dropout_p=args.dropout_p
            ),
            batch_size=args.batch_size,
            checkpoints_dir=args.checkpoints_dir,
            checkpoint_name='_'.join(
                [
                    args.model_name,
                    args.model_dir.split('/')[-1],
                    'drop' + str(args.dropout_p)
                ]
            ),
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            lr=args.lr
        )
    else:
        raise RuntimeError('Please check args.data_mode!!!')

    if args.mode == 'train':
        # default training from the beginning
        print('================= supertagging training =================\n')
        trainer.train()

    elif args.mode == 'train_on':
        # train from (checkpoint_epoch + 1)
        print('================= supertagging training on =================\n')
        trainer.load_checkpoint_and_train(checkpoint_epoch=args.checkpoint_epoch)
    
    elif args.mode == 'test':
        # test using the saved model from checkpoint_epoch
        print(f'================= supertagging testing: {args.test_mode} =================\n')
        trainer.load_checkpoint_and_test(
            checkpoint_epoch=args.checkpoint_epoch,
            mode=args.test_mode,
            data_mode=args.data_mode
        )

    else:
        raise RuntimeError('Please check the mode of the trainer!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='supertagging training')

    parser.add_argument('--sample_data_path', type=str,
                        default='../data/ccg-sample.auto')
    parser.add_argument('--ccgbank_train_data_path', type=str, default='../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--ccgbank_dev_data_path', type=str, default='../data/ccgbank-wsj_00.auto')
    parser.add_argument('--ccgbank_test_data_path', type=str, default='../data/ccgbank-wsj_23.auto')
    parser.add_argument('--treebanks_train_data_paths', type=str, nargs='+',
                        default=[
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
                        ], help='The list of paths to train data of treebanks')
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
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--model_name', type=str,
                        default='fc', choices=['fc', 'lstm', 'lstm-crf']) # Not used
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_lstm_layers', type=int, default=1) # Not used
    parser.add_argument('--dropout_p', type=float, default=0.5)

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'train_on', 'test'])
    parser.add_argument('--test_mode', type=str, default='dev_eval',
                        choices=['train_eval', 'dev_eval', 'test_eval'])
    parser.add_argument('--data_mode', type=str, default='treebanks',
                        choices=['treebanks', 'ccgbank'], help='Whether to use data of treebanks or CCGBank')
    parser.add_argument('--load_mode', type=str, default='first',
                        choices=['first', 'reuse'], help='Whether to load data from scratch or saved files')
    parser.add_argument('--checkpoint_epoch', type=int,
                        default=-1, help='The specific epoch of the checkpoint file to start with when training on, or to use when testing')

    args = parser.parse_args()
    main(args)
