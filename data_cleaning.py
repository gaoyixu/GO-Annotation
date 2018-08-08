"""Cleaning data.
Separate symbols, such as dots and commas.

Author:
    Yixu Gao
    gaoyixu1996@outlook.com

Usage:
    cleaning('data/term_name_def_descriptions_human.txt',
         'data/data_clean_lower.txt', True)
    cleaning('data/term_name_def_descriptions_human.txt',
         'data/data_clean.txt', False)
    cleaning('data/go_dict.txt', 'data/go_dict_clean_lower.txt', True)
    cleaning('data/go_dict.txt', 'data/go_dict_clean.txt', False)
    cleaning('data/gene_dict.txt', 'data/gene_dict_clean_lower.txt', True)
    cleaning('data/gene_dict.txt', 'data/gene_dict_clean.txt', False)
    narrow_vocabulary(['data/go_dict_clean_lower.txt',
                   'data/gene_dict_clean_lower.txt'],
                  'data/all_word_frequency.txt')
    narrow_word2vec('data/all_word_frequency.txt',
                    'data/glove.6B.200d.txt',
                    'data/glove.6B.200d.frequency_more_than_3.txt')
"""


def cleaning(src_path, dst_path, lower_mode=True):
    """Clean data.
    Separate symbols, such as dots and commas

    Args:
        src_path: string, e.g. 'go_dict.txt'
        dst_path: string, e.g. 'go_dict_clean.txt'
        lower_mode: bool
    """
    with open(src_path) as fd:
        lines = fd.readlines()
    letter_set = {'>', '_', '<', '*', '%', '=', '.', '&', '/',
                  '[', ']', '+', ')', '~', ':', '(', ';', ',', "'", '\\'}
    with open(dst_path, 'w') as fd:
        for line in lines:
            items = line.split('\t')
            item_cleaned_list = []
            for s in items:
                word_list = []
                word = ''
                for c in s:
                    if c.isalpha() or c.isdigit() or c == '-':
                        if lower_mode:
                            word += c.lower()
                        else:
                            word += c
                    else:
                        if word:
                            word_list.append(word)
                            word = ''
                        if c in letter_set:
                            word_list.append(c)
                if word:
                    word_list.append(word)
                item_cleaned_list.append(' '.join(word_list))
            fd.writelines('\t'.join(item_cleaned_list) + '\n')


def narrow_vocabulary(data_path_list, word_frequency_path):
    """Calculate word frequency of given files.

    Args:
        data_path_list: string list,
            e.g. ['go_dict_clean_lower.txt', 'gene_dict_clean_lower.txt']
        word_frequency_path: string, e.g. 'data/all_word_frequency.txt'
    """
    word_dict = {}
    for data_path in data_path_list:
        with open(data_path) as fd:
            lines = fd.readlines()
        for line in lines:
            words = line.split('\t')[1].split()
            for word in words:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    sorted_word_list = sorted(word_dict.items(), key=lambda x: x[1])
    with open(word_frequency_path, 'w') as fd:
        for word, frequency in sorted_word_list:
            fd.write(word + '\t' + str(frequency) + '\n')


def narrow_word2vec(word_frequency_path, word2vec_path, narrowed_path):
    """Narrow word2vec size according to word frequency.

    Args:
        word_frequency_path: string, e.g. 'all_word_frequency.txt'
        word2vec_path: string, e.g. 'glove.6B.200d.txt'
        narrowed_path: string, e.g. 'narrow_out.txt'
    """
    with open(word2vec_path) as fd:
        lines = fd.readlines()
    word_embedding_dict = {}
    for line in lines:
        items = line.split()
        word_embedding_dict[items[0]] = items[1:]
    with open(word_frequency_path) as fd:
        lines = fd.readlines()
    count = 0
    with open(narrowed_path, 'w') as fd:
        for line in lines:
            word, frequency = line.split()
            if int(frequency) > 3:
                if word in word_embedding_dict:
                    fd.write(word + ' ' + ' '.join(word_embedding_dict[word])
                             + '\n')
                    count += 1
    print(count)
