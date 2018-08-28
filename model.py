import tensorflow as tf

import os


from load_data import DataSet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_EPOCHS = 1


class Model:
    def __init__(self, hyperparameters=None,
                 train_set=None,
                 test_set=None):
        self.hyperparameters = hyperparameters
        self.train_set = train_set
        self.test_set = test_set

    def graph(self):

        pass

    def train(self):
        ds = DataSet(128, 'data/word_embedding.txt')
        train_dataset, test_dataset = ds.load_dict_data(
            'data/gene_dict_clean_lower.txt')
        iterator = train_dataset.make_initializable_iterator()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            for i in range(NUM_EPOCHS):
                sess.run(iterator.initializer)
                try:
                    # while True:
                        print(sess.run(iterator.get_next()))
                except tf.errors.OutOfRangeError:
                    pass

    def evaluate(self):
        pass


model = Model()
model.train()
