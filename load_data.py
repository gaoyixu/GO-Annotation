import numpy as np
import os


class DataSet:
    def __init__(self, word_embedding_path=''):
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

    def load_data_set(self):
        


data_set = DataSet('data/glove.6B.200d.txt')
data_set.load_word2vec()
print(data_set.word_embedding_dict.popitem())