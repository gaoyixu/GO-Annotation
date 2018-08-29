import tensorflow as tf

import os


import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_EPOCHS = 1


class Model:
    def __init__(self,
                 max_length,
                 encoder_hidden_sizes,
                 decoder_hidden_size,
                 batch_size):
        self.max_length = max_length
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_size = decoder_hidden_size
        self.batch_size = batch_size
        pass

    def graph(self):
        tf.get_variable('')
        pass

    def _encoder(self, batch_data):
        with tf.name_scope('encoder'):
            fw_cells = [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(size))
                      for size in self.encoder_hidden_sizes]
            bw_cells = [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(size))
                      for size in self.encoder_hidden_sizes]
            encoder_outputs_all, encoder_state_fw, encoder_state_bw = (
                tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    fw_cells, bw_cells, batch_data, dtype=tf.float32))
            encoder_outputs = tf.concat(encoder_outputs_all, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
        return encoder_outputs, encoder_state

    def _decoder(self, batch_data, encoder_outputs, encoder_state, training_mode=True):
        with tf.name_scope('decoder'), tf.variable_scope("decoder") as decoder_scope:
            if training_mode:
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    tf.nn.rnn_cell.BasicLSTMCell(self.decoder_hidden_size * 2),
                    tf.contrib.seq2seq.BahdanauAttention(
                        self.decoder_hidden_size * 2,
                        encoder_outputs,
                        normalize=True),
                    attention_layer_size=self.decoder_hidden_size * 2)
                initial_state = decoder_cell.zero_state(
                    dtype=tf.float32, batch_size=self.batch_size)
                initial_state = initial_state.clone(cell_state=encoder_state)
                helper = tf.contrib.seq2seq.TrainingHelper(batch_data, self.max_length)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=decoder_scope)
                self.decoder_output = outputs.rnn_output
                self.logits = self.projection_layer(self.decoder_output)
                self.logits_reshape = tf.concat(
                    [self.logits,
                     tf.zeros([self.batch_size, summary_max_len - tf.shape(self.logits)[1], self.vocabulary_size])],
                    axis=1)
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
                    decoder, output_time_major=True, maximum_iterations=summary_max_len, scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

    def _loss(self):
        with tf.name_scope("loss"):
            if not forward_only:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, summary_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(crossent * weights / tf.to_float(self.batch_size))

                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def train(self):
        with tf.name_scope('embedding'):
            word_vocab_list, embedding_matrix = (
                load_data.load_word_embedding_for_dict_file(
                    'data/gene_dict_clean_lower.txt',
                    'data/word_embedding.txt'))
            index_table = tf.contrib.lookup.index_table_from_tensor(
                word_vocab_list,
                num_oov_buckets=1,
                default_value=-1)
            total_words_num = len(word_vocab_list)

        with tf.name_scope('dataset'):
            ds = load_data.DataSet(
                'data/gene_dict_clean_lower.txt', self.max_length)
            train_dataset, test_dataset = ds.load_dict_data()

        iterator = train_dataset.make_initializable_iterator()

        with tf.Session() as sess:
            tf.summary.FileWriter('graphs/test', sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            for i in range(NUM_EPOCHS):
                sess.run(iterator.initializer)
                try:
                    # while True:
                        current_str = iterator.get_next()
                        current = index_table.lookup(current_str)
                        print(sess.run(current_str))
                        print(sess.run(current))
                        print(sess.run(tf.nn.embedding_lookup(
                            embedding_matrix, current)))
                        print(sess.run(tf.one_hot(current, total_words_num)))
                except tf.errors.OutOfRangeError:
                    pass

    def evaluate(self):
        pass


model = Model(15, [256], 256, 32)
model.train()
