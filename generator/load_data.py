import numpy as numpy
import torch
import os

logging.basicConfig(filename='log\log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S %p',
                    level=20)

def load_data_clean_lower(
    file_path,
    max_gene_length=18
    max_gene_num=50,
    max_term_length=18,
    batch_size=5,
    train_div=0.8,
    simple_concat=False):
    with open(file_path) as f:
        lines = lines

def load_word_embedding(vocab, path):
    word_vocab_list = []
    embedding_matrix = np.array([], dtype=np.float32)
    try:
        with open(word_embedding_path) as f:
            lines = fd.readlines()
            word_dimension

# word_vocab_list, embedding_matrix = load_data_clean_lower