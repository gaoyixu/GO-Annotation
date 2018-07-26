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
            line_cleaned = ''
            for s in [term_name, term_def] + descriptions:
                word_list = []
                word = ''
                for c in s:
                    if c.isalpha() or c.isdigit():
                        word += c.lower()
                    elif word:
                        word_list.append(word)
                        word = ''
                    elif c in letter_set:
                        word_list.append(c)
                    else:
                        pass
                line_cleaned += ' '.join(word_list) + '\t'
            fd.writelines([line_cleaned])


cleaning('term_name_def_descriptions_human.txt', 'data_clean_lower.txt')
