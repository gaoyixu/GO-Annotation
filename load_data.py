"""Load data.

Author:
    Yixu Gao
    gaoyixu1996@outlook.com

Usage:
    word_vocab_list, embedding_matrix = (
        load_data.load_word_embedding_for_dict_file(
            'data/gene_dict_clean_lower.txt',
            'data/word_embedding.txt'))
    index_table = tf.contrib.lookup.index_table_from_tensor(
        word_vocab_list,
        num_oov_buckets=1,
        default_value=-1)
    total_words_num = len(word_vocab_list)
    ds = load_data.DataSet(
        'data/gene_dict_clean_lower.txt', self.max_length)
    train_dataset, test_dataset = ds.load_dict_data()
    iterator = full_dataset.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        current_str = iterator.get_next()
        current = index_table.lookup(current_str)
        print(sess.run(current_str))
        print(sess.run(current))
        print(sess.run(tf.nn.embedding_lookup(
            embedding_matrix, current)))
        print(sess.run(tf.one_hot(current, total_words_num)))
"""


import numpy as np
import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_word_embedding(vocab, word_embedding_path, sign_words=None):
    """Load word embedding.

    Args:
        vocab: string set, vocab set of the dataset
        word_embedding_path: string, word embedding file path,
            e.g. glove.6B.200d.txt
        sign_words: ['PAD', '<s>', '<\s>']

    Returns:
        word_vocab_list: ['PAD', 'UNK', ...]
        embedding_matrix: 2-D matrix with vocab_size rows and dim columns
    """
    if sign_words is None:
        sign_words = ['PAD', '<s>', '<\s>']

    word_vocab_list = []
    embedding_matrix = np.array([], dtype=np.float32)

    try:
        with open(word_embedding_path) as fd:
            lines = fd.readlines()
            word_dimension = len(lines[0].split()) - 1

            word_vocab_list.extend(sign_words)

            for _ in sign_words:
                embedding_matrix = np.append(
                    embedding_matrix, np.zeros(word_dimension))

            for line in lines:
                items = line.split()
                word = items[0]
                embedding = np.array(items[1:], dtype=np.float32)
                if word in vocab:
                    word_vocab_list.append(word)
                    embedding_matrix = np.append(
                        embedding_matrix, embedding)

        # For default unknown word
        embedding_matrix = np.append(
            embedding_matrix, np.random.uniform(
                -1.0, 1.0, word_dimension))

        embedding_matrix = np.reshape(
            embedding_matrix,
            [-1, word_dimension]).astype(np.float32)

        return word_vocab_list, embedding_matrix

    except OSError:
        current_path = os.path.abspath(__file__)
        abs_file_path = os.path.abspath(os.path.dirname(current_path)
                                        + os.path.sep + word_embedding_path)
        print('Could not read embedding file: ' + abs_file_path)
        exit(0)


def load_word_embedding_for_dict_file(file_path, word_embedding_path):
    """Load word embedding for file.

    Args:
        file_path: string
        word_embedding_path: string
    """
    vocab = set()
    with open(file_path) as fd:
        lines = fd.readlines()
        for line in lines:
            words = line.split()[1:]
            vocab.update(words)
    return load_word_embedding(vocab, word_embedding_path)


class DataSet:
    def __init__(self, file_path, max_length):
        """
        Args:
            file_path: string, e.g. 'gene_dict_clean_lower.txt'
            max_length: int
        """
        self.file_path = file_path
        self.max_length = max_length
        self.dataset = None
        self.dataset_size = 0

    def _split_train_test(self,
                          batch_size,
                          padded_shapes,
                          train_div=0.8):
        """Split dataset to training set and test set.

        Args:
            batch_size: int
            padded_shapes: int
            train_div: float in [0, 1]

        Returns:
            train_dataset, test_dataset
        """
        if self.dataset:
            train_size = int(train_div * self.dataset_size)
            self.dataset = (self.dataset
                            .padded_batch(batch_size, padded_shapes,
                                          b'PAD'))
            train_dataset = self.dataset.take(train_size)
            test_dataset = self.dataset.skip(train_size)
            return train_dataset, test_dataset

    def load_dict_data(self):
        """Load data for gene dict and go dict.

        Returns:
            tf.data.Dataset for each contains
                (GeneID, Gene Description (string))
                e.g. (1, b'alpha-1-b glycoprotein')
        """
        with tf.name_scope('dataset'):
            with open(self.file_path) as fd:
                lines = fd.readlines()
            self.dataset_size = len(lines)
            del lines
            self.dataset = tf.data.TextLineDataset(self.file_path)
            self.dataset = (self.dataset.map(
                lambda x: tf.sparse_tensor_to_dense(
                    tf.string_split([x], '\t'), '')[0])
                .map(lambda x: tf.sparse_tensor_to_dense(
                    tf.string_split([x[1]]), '')[0][:self.max_length - 2])
                .map(lambda x: tf.concat(
                    [tf.convert_to_tensor(['<s>'], dtype=tf.string),
                     x, tf.convert_to_tensor(['<\s>'], dtype=tf.string)], 0)))
            train_dataset, test_dataset = self._split_train_test(2, self.max_length)
        return train_dataset, test_dataset
