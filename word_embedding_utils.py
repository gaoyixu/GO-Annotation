"""Utils for word embedding.

Author:
    Yixu Gao
    gaoyixu1996@outlook.com

Usage:
    batch_generator(FILE_LIST, VOCAB_SIZE, SKIP_WINDOW, BATCH_SIZE)
    dictionary, index_dictionary = load_vocab('visualization/vocab.tsv')
"""

from collections import Counter

import os

import numpy as np

import utils


def read_dict_data(file_path_list):
    """Read dict data into a list of lists of tokens.

    Args:
        file_path_list: string list,
            e.g. ['gene_dict.txt', 'go_dict.txt']

    Returns:
        a sentences list:
        [['atp', 'binding', 'cassette', 'subfamily', 'a', 'member', '1'], ...]
    """
    sentences_list = []
    for file_path in file_path_list:
        with open(file_path) as fd:
            lines = fd.readlines()
            for line in lines:
                sentences = line.split('\t')[1:]
                for sentence in sentences:
                    sentences_list.append(sentence.split())
    return sentences_list


def build_vocab(sentences_list, vocab_size, visual_fld):
    """Build vocabulary of VOCAB_SIZE most frequent words and save it to
    visualization/vocab.tsv

    Args:
        sentences_list:
        vocab_size: int, e.g. 5000
        visual_fld: string, fold, e.g. 'visualization'

    Returns:
        dictionary: {'a': 3}
        index_dictionary: {3: 'a'}
    """
    words = [word for sentence in sentences_list for word in sentence]
    utils.safe_mkdir(visual_fld)
    with open(os.path.join(visual_fld, 'vocab.tsv'), 'w') as fd:
        dictionary = {}
        index_dictionary = {}
        count = [('UNK', -1)]
        count.extend(Counter(words).most_common(vocab_size - 1))
        for index, (word, _) in enumerate(count):
            dictionary[word] = index
            index_dictionary[index] = word
            fd.write(word + '\n')

        return dictionary, index_dictionary


def convert_words_to_index(sentences_list, dictionary):
    """Replace each word in the dataset with its index in the dictionary.

    Args:
        sentences_list: [['atp', ..., 'cassette'], ...]
        dictionary: {'a': 0}

    Returns:
        an int list: [[2, 6,..., 342], ...]
    """
    return [[dictionary[word]
             if word in dictionary else 0
             for word in sentence] for sentence in sentences_list]


def generate_sample(index_sentences, context_window_size=3):
    """Form training pairs according to the skip-gram model.

    Args:
        index_sentences: an int list, [[2, 6,..., 342], ...]
        context_window_size: int, Default: 3

    Returns:
        center, target: index, e.g. (3, 5)
    """
    for index_sentence in index_sentences:
        for i, center in enumerate(index_sentence):
            for target in index_sentence[max(0, i - context_window_size): i]:
                yield center, target
            for target in index_sentence[i + 1: i + context_window_size + 1]:
                yield center, target


def most_common_words(n):
    """Create n most frequent words list to visualize on TensorBoard.
    Save it to visualization/vocab_[num_visualize].tsv

    Args:
        n: int
    """
    with open(os.path.join('visualization', 'vocab.tsv')) as fd:
        words = fd.readlines()[:n]
        words = [word for word in words]
    save_path = os.path.join('visualization', 'vocab_' + str(n) + '.tsv')
    with open(save_path, 'w') as fd:
        for word in words:
            fd.write(word)


def batch_generator(file_path_list, vocab_size=5000,
                    skip_window=3, batch_size=128):
    """Generate a batch_size list of center and target words in skip-gram.

    Args:
        file_path_list: string list
        vocab_size: int
        skip_window: int
        batch_size: int

    Returns:
        center_batch, target_batch:
            index lists with length of batch_size
    """
    sentences_list = read_dict_data(file_path_list)
    dictionary, index_dictionary = build_vocab(
        sentences_list, vocab_size, 'visualization')
    index_sentences = convert_words_to_index(sentences_list, dictionary)
    del sentences_list
    single_gen = generate_sample(index_sentences, skip_window)
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for i in range(batch_size):
            center_batch[i], target_batch[i] = next(single_gen)
        yield center_batch, target_batch


def load_vocab(file_path):
    """Load visualization/vocab.tsv

    Args:
        file_path: string
    """
    dictionary = {}
    index_dictionary = {}
    with open(file_path) as fd:
        lines = fd.readlines()
        for i, word in enumerate(lines):
            word = word.strip()
            dictionary[word] = i
            index_dictionary[i] = word
    return dictionary, index_dictionary
