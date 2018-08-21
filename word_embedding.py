""" starter code for word2vec skip-gram model with NCE loss
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 04
"""

import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import utils
import word_embedding_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FILE_LIST = ['data/gene_dict_clean_lower.txt', 'data/go_dict_clean_lower.txt']
VOCAB_SIZE = 50000
BATCH_SIZE = 128
SKIP_WINDOW = 3
EMBED_SIZE = 128
NUM_SAMPLED = 500
LEARNING_RATE = 0.1
NUM_TRAIN_STEPS = 100000
SKIP_STEP = 5000
NUM_VISUALIZE = 3000


class SkipGramModel:
    """Build the graph for skip-gram model """

    def __init__(self, dataset, vocab_size, embed_size,
                 batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.get_variable('global_step',
                                           initializer=tf.constant(0),
                                           trainable=False)
        self.skip_step = SKIP_STEP
        self.dataset = dataset

    def _import_data(self):
        """Import data."""
        with tf.name_scope('data'):
            self.iterator = self.dataset.make_initializable_iterator()
            self.center_words, self.target_words = self.iterator.get_next()

    def _create_embedding(self):
        """Define weights and embedding lookup."""
        with tf.name_scope('embed'):
            self.embed_matrix = tf.get_variable(
                'embed_matrix', shape=[self.vocab_size, self.embed_size],
                initializer=tf.random_uniform_initializer())
            self.embed = tf.nn.embedding_lookup(
                self.embed_matrix, self.center_words, name='embedding')

    def _create_loss(self):
        """Define the loss function."""
        with tf.name_scope('loss'):
            # construct variables for NCE loss
            nce_weight = tf.get_variable(
                'nce_weight',
                shape=[self.vocab_size, self.embed_size],
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / (self.embed_size ** 0.5)))
            nce_bias = tf.get_variable('nce_bias',
                                       initializer=tf.zeros([VOCAB_SIZE]))

            # define loss function to be NCE loss function
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weight,
                               biases=nce_bias,
                               labels=self.target_words,
                               inputs=self.embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        """Define optimizer."""
        self.optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss,
                                         global_step=self.global_step)

    def _create_summaries(self):
        """Define summaries."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """Build the graph for our model."""
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_train_steps):
        saver = tf.train.Saver()

        utils.safe_mkdir('checkpoints')
        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter('graphs/word2vec/lr' + str(self.learning_rate), sess.graph)
            initial_step = self.global_step.eval()

            for index in range(initial_step, initial_step + num_train_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % self.skip_step == 0:
                        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / self.skip_step))
                        total_loss = 0.0
                        saver.save(sess, 'checkpoints/skip-gram', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()

    def visualize(self, visual_fld, num_visualize):
        """ run "'tensorboard --logdir='visualization'" to see the embeddings """

        # create the list of num_variable most common words to visualize
        word_embedding_utils.most_common_words(num_visualize)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            final_embed_matrix = sess.run(self.embed_matrix)

            # you have to store embeddings in a new variable
            embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            summary_writer = tf.summary.FileWriter(visual_fld)

            # add embedding to the config file
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # link this tensor to its metadata file, in this case the first NUM_VISUALIZE words of vocab
            embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

            # saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            saver_embed = tf.train.Saver([embedding_var])
            saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)


def generator():
    """Yield from batch generator in word_embedding_utils.

    Return:
        batch generator
    """
    yield from word_embedding_utils.batch_generator(
        FILE_LIST, VOCAB_SIZE, SKIP_WINDOW, BATCH_SIZE)


def main():
    dataset = tf.data.Dataset.from_generator(
        generator, (tf.int32, tf.int32),
        (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    model = SkipGramModel(dataset, VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    model.train(NUM_TRAIN_STEPS)
    model.visualize('visualization', NUM_VISUALIZE)


if __name__ == '__main__':
    main()
