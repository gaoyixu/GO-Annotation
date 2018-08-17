import numpy as np
import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataSet:
    def __init__(self, hyperparameters=None, word_embedding_path=''):
        self.hyperparameters = hyperparameters
        self.word_embedding_path = word_embedding_path
        self.word_embedding_dict = {}

    def load_word2vec(self, file_path=''):
        if not file_path:
            file_path = self.word_embedding_path
        else:
            self.word_embedding_path = file_path
        try:
            with open(file_path) as fd:
                lines = fd.readlines()
            for line in lines:
                items = line.split()
                self.word_embedding_dict[items[0]] = np.array(items[1:],
                                                              dtype=np.float64)
        except OSError:
            current_path = os.path.abspath(__file__)
            abs_file_path = os.path.abspath(os.path.dirname(current_path)
                                            + os.path.sep + file_path)
            print('Could not read embedding file: ' + abs_file_path)
            exit(0)

    def get_word_embedding(self, word):
        if not self.word_embedding_dict:
            print('Please Load Legal Word Embeddings First!')
            exit(0)
        if word in self.word_embedding_dict:
            return self.word_embedding_dict[word]
        else:
            return np.zeros(self.hyperparameters.word_dimension)

    @staticmethod
    def load_dict_data(file_path):
        """Load data for gene dict and go dict.

        Args:
            file_path: string, e.g. 'gene_dict_clean_lower.txt'

        Returns:
            tf.data.Dataset for each contains
                (GeneID, Gene Description (string))
                e.g. (1, b'alpha-1-b glycoprotein')
        """
        dataset = tf.data.TextLineDataset(file_path)
        dataset = dataset.map(lambda x: tf.string_split([x], '\t'))
        dataset = dataset.map(lambda x: tf.sparse_tensor_to_dense(x, '')[0])
        dataset = dataset.map(lambda x: (x[0], '<start> ' + x[1] + ' <end>'))
        return dataset


# data_set = DataSet('data/glove.6B.200d.frequency_more_than_3.txt')
# data_set.load_word2vec()
# print(data_set.word_embedding_dict.popitem())
full_dataset = DataSet().load_dict_data('data/gene_dict_clean_lower.txt')
iterator = full_dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    print(sess.run(one_element))
