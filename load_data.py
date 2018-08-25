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
        self.word_embedding_dict = {}
        self.word_vocab_list = []
        self.embedding_matrix = np.array([], dtype=np.float32)
        self.int_to_vocab = {}
        self.vocab_to_int = {}
        self.word_embedding = None

    def load_word_embedding(self, vocab, file_path=''):
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
            word_embedding: an tensor constant
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
                    np.append(self.embedding_matrix, np.random.uniform(
                        -1.0, 1.0, self.word_dimension))

                for line in lines:
                    items = line.split()
                    word = items[0]
                    embedding = np.array(items[1:], dtype=np.float32)
                    if word in vocab:
                        self.word_embedding_dict[word] = embedding
                        self.word_vocab_list.append(word)
                        np.append(self.embedding_matrix, embedding)

            self.embedding_matrix = np.reshape(
                self.embedding_matrix,
                [-1, self.word_dimension]).astype(np.float32)

            for i, word in enumerate(self.word_vocab_list):
                self.int_to_vocab[i] = word
                self.vocab_to_int[word] = i

            self.word_embedding = self.embedding_matrix

            return (self.word_vocab_list, self.embedding_matrix,
                    self.int_to_vocab, self.vocab_to_int, self.word_embedding)

        except OSError:
            current_path = os.path.abspath(__file__)
            abs_file_path = os.path.abspath(os.path.dirname(current_path)
                                            + os.path.sep + file_path)
            print('Could not read embedding file: ' + abs_file_path)
            exit(0)

    def get_word_embedding_from_dict(self, word):
        """Get word embedding.
        Note that you should load word embedding first!

        Args:
            word: string

        Returns:
            word embedding: 1-D np.array of word_dimension length
        """
        if not self.word_embedding_dict:
            print('Please Load Legal Word Embeddings First!')
            exit(0)
        if word in self.word_embedding_dict:
            return self.word_embedding_dict[word]
        else:
            if 'UNK' in self.word_embedding_dict:
                return self.word_embedding_dict['UNK']
            else:
                return np.zeros(self.word_dimension)

    def _encode_sentence_to_index(self, sentence):
        """Encode the sentence using vocab.

        Args:
            sentence: tf.string

        Returns:
            encoded sentence: 1-D int list of word index
        """
        words = tf.sparse_tensor_to_dense(tf.string_split([sentence]), '')[0]
        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                keys=self.word_vocab_list,
                values=tf.convert_to_tensor(
                    np.array(range(len(self.word_vocab_list)))))
            , self.vocab_to_int['UNK'])
        encoded_sentence = table.lookup(words)
        return encoded_sentence

    def _encode_indexed_sentence_to_embedding(self, indexed_sentence):
        """Encode the sentence using vocab.

        Args:
            indexed_sentence: 1-D int list of word index

        Returns:
            encoded sentence: 2-D tensor of word embedding
        """
        return tf.nn.embedding_lookup(self.word_embedding,
                                      indexed_sentence)

    def _encode_sentence_to_embedding(self, index, sentence):
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
        vocab = set()
        with open(file_path) as fd:
            lines = fd.readlines()
            for line in lines:
                words = line.split()[1:]
                vocab.update(words)
        self.load_word_embedding(vocab)
        dataset = tf.data.TextLineDataset(file_path)
        dataset = dataset.map(lambda x: tf.string_split([x], '\t'))
        dataset = dataset.map(lambda x: tf.sparse_tensor_to_dense(x, '')[0])
        dataset = dataset.map(lambda x: (x[0], '<s> ' + x[1] + ' <\s>'))
        dataset = dataset.map(self._encode_sentence_to_embedding)
        return dataset


# data_set = DataSet('data/glove.6B.200d.frequency_more_than_3.txt')
# data_set.load_word2vec()
# print(data_set.word_embedding_dict.popitem())
dataset = DataSet(128, 'data/word_embedding.txt')
full_dataset = dataset.load_dict_data('data/gene_dict_clean_lower.txt')
iterator = full_dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    print(sess.run(one_element))
