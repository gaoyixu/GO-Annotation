def cleaning(src_path, dst_path):
    with open(src_path) as fd:
        lines = fd.readlines()
    letter_set = {'>', '_', '<', '-', '*', '%', '=', '.', '&', '/',
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
            word_list = ''
            for s in [term_name, term_def] + descriptions:
                word_list = []
                word = ''
                for c in s:
                    if c.isalpha() or c.isdigit():
                        word += c  # .lower()
                    elif word:
                        word_list.append(word)
                        word = ''
                    elif c in letter_set:
                        word_list.append(c)
                    else:
                        pass
                item_cleaned_list.append(' '.join(word_list))
            fd.writelines('\t'.join(item_cleaned_list) + '\n')


cleaning('term_name_def_descriptions_human.txt', 'data_clean.txt')
