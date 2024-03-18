import sys
import re
import torch
from typing import List, Dict, Tuple

sys.path.append('..')
from data_loader import DataItem, load_auto_file


DATA_MASK_PADDING = 0
TARGET_PADDING = -100


def pre_tokenize_sent(sent: str) -> List[str]:
    # This function is not very complete yet!!!
    splited = sent.split(' ')
    returned = list()
    for token in splited:
        if re.match('[a-zA-Z]', token[-1]):
            returned.append(token)
        elif token == 'Mr.' or token == 'Ms.':
            returned.append(token)
        else:
            returned.extend([token[0: -1], token[-1]])
    return returned


def get_cat_ids(
    categories: List[str],
    category2idx: Dict[str, int]
) -> List[int]:
    return [
        category2idx[category] if category in category2idx else TARGET_PADDING
        for category in categories
    ]


def prepare_data(
    data_items: List[DataItem],
    tokenizer,
    category2idx: Dict[str, int]
):
    """
    Output: wrapped data needed to input into the model
    """
    data = list()  # a list containing a list of input_ids for each sentence
    mask = list()  # a list containing the attention mask list for each sentence
    word_piece_tracked = list()  # a list containing the list of word_piece_tracked for each sentence
    target = list()  # a list containing the list of one-hot vectors for each sentence
    ids = list() # a list containing the ids of each sentence, e.g., filename_id_str

    for data_item in data_items:
        ids.append(data_item.filename + '_' + data_item.id)
        pretokenized_sent = [token.contents for token in data_item.tokens]
        
        try:
            categories = [str(token.tag) for token in data_item.tokens]
        except:
            continue

        word_piece_tracked.append(
            [
                len(item)
                for item in tokenizer(pretokenized_sent, add_special_tokens=False).input_ids
            ]
        )

        inputs = tokenizer(
            pretokenized_sent,
            add_special_tokens=False,
            is_split_into_words=True,
            truncation=True
        )
        data.append(inputs.input_ids)
        mask.append(inputs.attention_mask)
        target.append(get_cat_ids(categories, category2idx))

    max_length = max(
        max([len(input_ids) for input_ids in data]),
        max([len(tgt) for tgt in target])
    )
    for i in range(len(data)):
        assert len(data[i]) == len(mask[i])
        data[i] = data[i] + [DATA_MASK_PADDING] * (max_length - len(data[i]))  # padding
        mask[i] = mask[i] + [DATA_MASK_PADDING] * (max_length - len(mask[i]))  # padding
    for i in range(len(target)):
        target[i] = target[i] + [TARGET_PADDING] * (max_length - len(target[i]))  # padding

    return {
        'ids': ids,
        'input_ids': torch.LongTensor(data),
        'mask': torch.FloatTensor(mask),
        'word_piece_tracked': word_piece_tracked,
        'target': torch.LongTensor(target)
    }


def calculate_treebanks_statistics(
    treebanks_data_path: List[str]
) -> List[Tuple[str, int, int, int]]:
    """
    Outputs: A sorted list of dictionaries, each of which maps the treebank name to a tuple
        storing (treebank name, total number of sentences, total number of tokens, average number of tokens per sent),
        sorted from high to low, according to the number of tokens.
    """
    treebanks_statistics = list()
    for path in treebanks_data_path:
        treebank_name = path.split('/')[-2]
        n_tokens = 0
        
        try:
            data_items, _ = load_auto_file(filename=path)
        except:
            continue
        n_sents = len(data_items)
        if n_sents == 0:
            raise Exception('Empty file:', path)
        for data_item in data_items:
            n_tokens += len(data_item.tokens)

        treebanks_statistics.append((treebank_name, n_sents, n_tokens, round(n_tokens / n_sents, 2)))
    
    return sorted(treebanks_statistics, key=lambda x:x[2], reverse=True)


def results_printer(
    train_treebanks_statistics: List[Tuple[str, int, int, int]],
    test_treebanks_statistics: List[Tuple[str, int, int, int]],
    evaluation_results: Dict[str, Tuple[float, float]] # (loss, acc)
): # Print supertagging results in the order of treebank size (from high to low)
    print('============================== Printing Evaluation Results in the Order of Training Treebank Size ==============================\n')
    for train_treebank in train_treebanks_statistics:
        name, train_n_sents, train_n_tokens, train_ave_token_per_sent = train_treebank
        for test_treebank in test_treebanks_statistics:
            if name == test_treebank[0]:
                _, test_n_sents, test_n_tokens, test_ave_token_per_sent = test_treebank
                print(f'[{name}] || Training size ~ n_sents: {train_n_sents}, n_tokens: {train_n_tokens}, ave_token_per_sent: {train_ave_token_per_sent} || Testing size ~ n_sents: {test_n_sents}, n_tokens: {test_n_tokens}, ave_token_per_sent: {test_ave_token_per_sent} || ave_loss: {evaluation_results[name][0]}, acc: {evaluation_results[name][1]}')