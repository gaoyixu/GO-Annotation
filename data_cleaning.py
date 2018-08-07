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
"""


def cleaning(src_path, dst_path, lower_mode=True):
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


cleaning('data/term_name_def_descriptions_human.txt',
         'data/data_clean_lower.txt', True)