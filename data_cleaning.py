def cleaning(src_path, dst_path):
    with open(src_path) as fd:
        lines = fd.readlines()
    letter_set = {'>', '_', '<', '*', '%', '=', '.', '&', '/',
                  '[', ']', '+', ')', '~', ':', '(', ';', ',', "'", '\\'}
    with open(dst_path, 'w') as fd:
        for line in lines:
            items = line.split('\t')
            term_name = items[0]
            items.pop(0)
            term_def = items[0]
            items.pop(0)
            descriptions = items
            item_cleaned_list = []
            for s in [term_name, term_def] + descriptions:
                word_list = []
                word = ''
                for c in s:
                    if c.isalpha() or c.isdigit() or c == '-':
                        word += c.lower()
                    else:
                        if word:
                            word_list.append(word)
                            word = ''
                        if c in letter_set:
                            word_list.append(c)
                item_cleaned_list.append(' '.join(word_list))
            fd.writelines('\t'.join(item_cleaned_list) + '\n')


def clean_dict(src_path, dst_path):
    with open(src_path) as fd:
        lines = fd.readlines()
    letter_set = {'>', '_', '<', '*', '%', '=', '.', '&', '/',
                  '[', ']', '+', ')', '~', ':', '(', ';', ',', "'", '\\'}
    with open(dst_path, 'w') as fd:
        for line in lines:
            items = line.split('\t')
            item_cleaned_list = []
            for s in items[1:]:
                word_list = []
                word = ''
                for c in s:
                    if c.isalpha() or c.isdigit() or c == '-':
                        word += c  # .lower()
                    else:
                        if word:
                            word_list.append(word)
                            word = ''
                        if c in letter_set:
                            word_list.append(c)
                if word:
                    word_list.append(word)
                item_cleaned_list.append(' '.join(word_list))
            fd.writelines(items[0] + '\t' + '\t'.join(item_cleaned_list) + '\n')


clean_dict('go_dict.txt', 'go_dict_clean.txt')
# cleaning('term_name_def_descriptions_human.txt', 'data_clean_lower.txt')
