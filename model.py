import tensorflow as tf


class Model:
    def __init__(self, hyperparameters, train_set, test_set):
        self.hyperparameters = hyperparameters
        self.train_set = train_set
        self.test_set = test_set

    def graph(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass
