import tensorflow as tf

import os


import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_EPOCHS = 1


class Model:
    def __init__(self,
                 max_length,
                 embedding_dimension,
                 encoder_hidden_sizes,
                 decoder_hidden_size,
                 batch_size):
        self.max_length = max_length
        self.embedding_dimension = embedding_dimension
        self.vocab_size = 0
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_size = decoder_hidden_size
        self.batch_size = batch_size

        self.loss = None
        self.update = None

    def graph(self, batch_data, embedded_batch_data,
              one_hot_batch_data, decoder_length):
        encoder_outputs, encoder_state = self._encoder(embedded_batch_data)
        logits = self._decoder(embedded_batch_data, encoder_outputs, encoder_state, decoder_length)
        loss = self._loss(logits, batch_data, decoder_length)
        self._optimizer(loss)

    def _encoder(self, embedded_batch_data):
        with tf.name_scope('encoder'):
            fw_cells = [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(size))
                      for size in self.encoder_hidden_sizes]
            initial_states_fw = [cell.zero_state(
                self.batch_size, dtype=tf.float32)
                                 for cell in fw_cells]
            bw_cells = [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(size))
                for size in self.encoder_hidden_sizes]
            initial_states_bw = [cell.zero_state(
                self.batch_size, dtype=tf.float32)
                for cell in bw_cells]
            encoder_outputs_all, encoder_state_fw, encoder_state_bw = (
                tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    fw_cells, bw_cells, embedded_batch_data,
                    initial_states_fw=initial_states_fw,
                    initial_states_bw=initial_states_bw,
                    dtype=tf.float32))
            encoder_outputs = tf.concat(encoder_outputs_all, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c,
                                         encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h,
                                         encoder_state_bw[0].h), 1)
            encoder_state = tf.contrib.rnn.LSTMStateTuple(
                c=encoder_state_c, h=encoder_state_h)
        return encoder_outputs, encoder_state

    def _decoder(self,
                 batch_data,
                 encoder_outputs,
                 encoder_state,
                 decoder_length,
                 training_mode=True):
        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(self.vocab_size,
                                                    use_bias=False)

        with tf.name_scope('decoder'), tf.variable_scope('decoder') as decoder_scope:
            if training_mode:
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    tf.nn.rnn_cell.BasicLSTMCell(self.decoder_hidden_size * 2),
                    tf.contrib.seq2seq.BahdanauAttention(
                        self.decoder_hidden_size * 2,
                        encoder_outputs, normalize=True),
                    attention_layer_size=self.decoder_hidden_size * 2)
                initial_state = decoder_cell.zero_state(
                    dtype=tf.float32, batch_size=self.batch_size)
                initial_state = initial_state.clone(cell_state=encoder_state)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    batch_data, tf.cast(decoder_length, tf.int32))
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, scope=decoder_scope)
                decoder_output = outputs.rnn_output
                logits = self.projection_layer(decoder_output)
                logits_reshape = tf.concat(
                    [logits,
                     tf.zeros([self.batch_size,
                               self.max_length - tf.shape(logits)[1],
                               self.vocab_size])],
                    axis=1)
                return logits_reshape

            else:
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state,
                                                                          multiplier=self.beam_width)
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.X_len, multiplier=self.beam_width)
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.num_hidden * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
                initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=self.max_length, scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

    def _loss(self, logits, decoder_target, decoder_length):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=decoder_target)
            weights = tf.sequence_mask(decoder_length, self.max_length, dtype=tf.float32)
            self.loss = tf.reduce_sum(cross_entropy * weights / tf.to_float(self.batch_size))
        return self.loss

    def _optimizer(self, loss):
        with tf.name_scope('optimizer'):
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            optimizer = tf.train.AdamOptimizer(0.01)
            self.update = optimizer.apply_gradients(
                zip(clipped_gradients, params),
                global_step=tf.Variable(0, trainable=False))

    def train(self):
        with tf.name_scope('dataset'):
            (train_input_data,
             train_output_data,
             test_input_data,
             test_output_data) = load_data.load_data_clean_lower(
                'data/data_clean_lower.txt', 18, 10, simple_concat=True)

        with tf.name_scope('embedding'):
            word_vocab_list, embedding_matrix = (
                load_data.load_word_embedding_for_dict_file(
                    'data/gene_dict_clean_lower.txt',
                    'data/word_embedding.txt'))

            index_table = tf.contrib.lookup.index_table_from_tensor(
                word_vocab_list,
                num_oov_buckets=1,
                default_value=-1)
            self.vocab_size = len(word_vocab_list)
            print()

        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            for i in range(len(train_input_data)):
                input_batch_data = index_table.lookup(
                    tf.convert_to_tensor(train_input_data[i]))
                output_batch_data = index_table.lookup(
                    tf.convert_to_tensor(train_output_data[i]))
                decoder_length = tf.where(tf.equal(output_batch_data, 2))[:, -1] + 1
                embedded_input_batch_data = tf.nn.embedding_lookup(
                    embedding_matrix, input_batch_data)
                embedded_output_batch_data = tf.nn.embedding_lookup(
                    embedding_matrix, output_batch_data)
                one_hot_input_batch_data = tf.one_hot(
                    input_batch_data, self.vocab_size)
                one_hot_input_batch_data = tf.one_hot(
                    output_batch_data, self.vocab_size)

                self.graph(batch_data,
                           embedded_batch_data,
                           one_hot_batch_data,
                           decoder_length)

                sess.run(tf.global_variables_initializer())
                tf.summary.FileWriter('graphs/test', sess.graph)

                for j in range(NUM_EPOCHS):
                    try:
                        a = sess.run(self.update)
                        b = sess.run(self.loss)
                        print(a)
                        print(b)

                    except tf.errors.OutOfRangeError:
                        pass

    def test(self):
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())

            batch_data = index_table.lookup(iterator.get_next())
            decoder_length = tf.where(tf.equal(batch_data, 2))[:, -1] + 1
            embedded_batch_data = tf.nn.embedding_lookup(
                embedding_matrix, batch_data)
            one_hot_batch_data = tf.one_hot(batch_data, self.vocab_size)


def main(used_argv):
    model = Model(15, 128, [256], 256, 32)
    model.train()


if __name__ == '__main__':
    tf.app.run(main)
