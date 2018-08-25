"""Load data.

Author:
    Yixu Gao
    gaoyixu1996@outlook.com

Usage:
    ds = DataSet(128, 'data/word_embedding.txt')
    full_dataset = ds.load_dict_data('data/gene_dict_clean_lower.txt')
    iterator = full_dataset.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print(sess.run(iterator.get_next()))
"""


import numpy as np
import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SIGN_WORDS = ['PAD', 'UNK', '<s>', '<\s>']


class DataSet:
    def __init__(self,
                 word_dimension,
                 word_embedding_path,
                 sign_words=SIGN_WORDS):
        self.word_embedding_path = word_embedding_path
        self.word_dimension = word_dimension
        self.sign_words = sign_words
        self.word_vocab_list = []
        self.embedding_matrix = np.array([], dtype=np.float32)
        self.int_to_vocab = {}
        self.vocab_to_int = {}
        self.table = None

    def _load_word_embedding(self, vocab, file_path=''):
        """Load word embedding.

        Args:
            vocab: string set, vocab set of the dataset
            file_path: string, word embedding file path,
                e.g. glove.6B.200d.txt

        Returns:
            word_vocab_list: ['PAD', 'UNK', ...]
            embedding_matrix: 2-D matrix with vocab_size rows and dim columns
            int_to_vocab: {0: 'PAD',1: 'UNK', ...}
            vocab_to_int: {'PAD': 0, 'UNK': 1, ...}
        """
        if not file_path:
            file_path = self.word_embedding_path
        else:
            self.word_embedding_path = file_path
        try:
            with open(file_path) as fd:
                lines = fd.readlines()

                self.word_vocab_list.extend(self.sign_words)
                for _ in self.sign_words:
                    self.embedding_matrix = np.append(
                        self.embedding_matrix, np.random.uniform(
                            -1.0, 1.0, self.word_dimension))

                for line in lines:
                    items = line.split()
                    word = items[0]
                    embedding = np.array(items[1:], dtype=np.float32)
                    if word in vocab:
                        self.word_vocab_list.append(word)
                        self.embedding_matrix = np.append(
                            self.embedding_matrix, embedding)

            self.embedding_matrix = np.reshape(
                self.embedding_matrix,
                [-1, self.word_dimension]).astype(np.float32)

            for i, word in enumerate(self.word_vocab_list):
                self.int_to_vocab[i] = word
                self.vocab_to_int[word] = i

            self.table = tf.contrib.lookup.index_table_from_tensor(
                self.word_vocab_list,
                num_oov_buckets=1,
                default_value=self.vocab_to_int['UNK'])

            return (self.word_vocab_list, self.embedding_matrix,
                    self.int_to_vocab, self.vocab_to_int)

        except OSError:
            current_path = os.path.abspath(__file__)
            abs_file_path = os.path.abspath(os.path.dirname(current_path)
                                            + os.path.sep + file_path)
            print('Could not read embedding file: ' + abs_file_path)
            exit(0)

    def _load_word_embedding_for_file(self, file_path):
        """Load word embedding for file.

        Args:
            file_path: string
        """
        vocab = set()
        with open(file_path) as fd:
            lines = fd.readlines()
            for line in lines:
                words = line.split()[1:]
                vocab.update(words)
        self._load_word_embedding(vocab)

    def _encode_sentence_to_index(self, sentence):
        """Encode the sentence using vocab.

        Args:
            sentence: tf.string

        Returns:
            encoded sentence: 1-D int list of word index
        """
        words = tf.sparse_tensor_to_dense(tf.string_split([sentence]), '')[0]
        encoded_sentence = self.table.lookup(words)
        return encoded_sentence

    def _encode_indexed_sentence_to_embedding(self, indexed_sentence):
        """Encode the sentence using vocab.

        Args:
            indexed_sentence: 1-D int list of word index

        Returns:
            encoded sentence: 2-D tensor of word embedding
        """
        return tf.nn.embedding_lookup(
            self.embedding_matrix, indexed_sentence)

    def _encode_sentence_to_embedding(self, _, sentence):
        """Encode the sentence using vocab.

        Args:
            sentence: string

        Returns:
            encoded sentence: 2-D tensor of word embedding
        """
        indexed_sentence = self._encode_sentence_to_index(sentence)
        return self._encode_indexed_sentence_to_embedding(indexed_sentence)

    def load_dict_data(self, file_path):
        """Load data for gene dict and go dict.

        Args:
            file_path: string, e.g. 'gene_dict_clean_lower.txt'

        Returns:
            tf.data.Dataset for each contains
                (GeneID, Gene Description (string))
                e.g. (1, b'alpha-1-b glycoprotein')
        """
        self._load_word_embedding_for_file(file_path)
        dataset = tf.data.TextLineDataset(file_path)
        dataset = dataset.map(
            lambda x: tf.sparse_tensor_to_dense(
                tf.string_split([x], '\t'), '')[0])
        dataset = dataset.map(lambda x: (x[0], '<s> ' + x[1] + ' <\s>'))
        dataset = dataset.map(self._encode_sentence_to_embedding)
        return dataset


ds = DataSet(128, 'data/word_embedding.txt')
full_dataset = ds.load_dict_data('data/gene_dict_clean_lower.txt')
iterator = full_dataset.make_initializable_iterator()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run(iterator.get_next()))
