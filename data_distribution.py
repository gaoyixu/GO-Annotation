"""Draw statistics figures of dataset.

Author:
    Yixu Gao
    gaoyixu1996@outlook.com

Usage:
    draw_term_name_len_distribution(
        'term_name_def_descriptions_human.txt')
    draw_term_def_distribution_less_than_100(
        'term_name_def_descriptions_human.txt')
    draw_term_def_distribution_more_than_100(
        'term_name_def_descriptions_human.txt')
    draw_term_gene_distribution(
        'term_name_def_descriptions_human.txt')
    draw_term_gene_length_distribution(
        'term_name_def_descriptions_human.txt')

    draw_word_frequency_distribution(
        'all_word_frequency.txt')
    show_words_not_in_vocabulary(
        'data/go_dict_clean_lower.txt',
        'data/glove.6B.50d.frequency_more_than_3.txt')
"""
from matplotlib import pyplot as plt
from numpy import mean, median


def draw_term_name_len_distribution(filename):
    """Draw distribution for term name length.

    Args:
        filename: 'term_name_def_descriptions_human.txt'
    """
    with open(filename) as fd:
        lines = fd.readlines()
    name_len_list = []
    for line in lines:
        items = line.split('\t')
        term_name = items[0]
        name_words_list = term_name.split()
        name_len_list.append(len(name_words_list))

    print(max(name_len_list), mean(name_len_list))

    n, bins, _ = plt.hist(name_len_list, bins=range(1, 30),
                          color='r', edgecolor='#FFFFFF')
    print(n, bins)
    max_i = 0
    for i in range(len(n) - 1, -1, -1):
        if n[i] > 0:
            max_i = i
            break
    print(bins[max_i])
    print(n[max_i])

    def bias(k):
        if k < 10:
            return 0.4
        elif k < 1000:
            return 0.25
        elif k < 10000:
            return 0.15

    for i in range(len(n)):
        plt.text(bins[i] + bias(n[i]), n[i] + 20, str(int(n[i])), fontsize=8)
    plt.title('Term Name Length Distribution')
    plt.xticks([i + 0.5 for i in range(1, 29)],
               [str(i) for i in range(1, 29)], fontsize=8)
    plt.xlabel('Number of Name Words')
    plt.ylabel('Number')
    plt.show()


def draw_term_def_distribution_less_than_100(filename):
    """Draw distribution for term define length (<100).

    Args:
        filename: 'term_name_def_descriptions_human.txt'
    """
    with open(filename) as fd:
        lines = fd.readlines()
    def_len_list = []
    for line in lines:
        items = line.split('\t')
        term_def = items[1]
        def_words_list = term_def.split()
        def_len_list.append(len(def_words_list))

    print(max(def_len_list), mean(def_len_list))

    n, bins, _ = plt.hist(def_len_list, bins=range(1, 100),
                          color='b', edgecolor='#FFFFFF')
    print(n, bins)
    max_i = 0
    for i in range(len(n) - 1, -1, -1):
        if n[i] > 0:
            max_i = i
            break
    print(bins[max_i])
    print(n[max_i])

    def bias(j):
        if j < 10:
            return 0.3
        elif j < 100:
            return 0.15
        elif j < 1000:
            return 0.05
        elif j < 10000:
            return -0.1

    for i in range(len(n)):
        plt.text(bins[i] + bias(n[i]), n[i] + 5, str(int(n[i])), fontsize=4)
    plt.title('Term Define Length (Less Than 100) Distribution')
    plt.axis([0, 100, 0, 1600])
    plt.xticks([i + 0.5 for i in range(0, 100, 5)],
               [str(i) for i in range(0, 100, 5)], fontsize=8)
    plt.xlabel('Number of Define Words')
    plt.ylabel('Number')
    plt.show()


def draw_term_def_distribution_more_than_100(filename):
    """Draw distribution for term define length (>100).

    Args:
        filename: 'term_name_def_descriptions_human.txt'
    """
    with open(filename) as fd:
        lines = fd.readlines()
    def_len_list = []
    for line in lines:
        items = line.split('\t')
        term_def = items[1]
        def_words_list = term_def.split()
        def_len_list.append(len(def_words_list))

    print(max(def_len_list), mean(def_len_list))

    n, bins, _ = plt.hist(def_len_list, bins=range(100, 200),
                          color='m', edgecolor='#FFFFFF')
    print(n, bins)
    max_i = 0
    for i in range(len(n) - 1, -1, -1):
        if n[i] > 0:
            max_i = i
            break
    print(bins[max_i])
    print(n[max_i])
    plt.title('Term Define Length (More Than 100) Distribution')
    plt.axis([100, 200, 0, 8])
    plt.xticks([i + 0.5 for i in range(100, 200, 5)],
               [str(i) for i in range(100, 200, 5)], fontsize=8)
    plt.xlabel('Number of Define Words')
    plt.ylabel('Number')
    plt.show()


def draw_term_gene_distribution(filename):
    """Draw distribution for gene number.

    Args:
        filename: 'term_name_def_descriptions_human.txt'
    """
    with open(filename) as fd:
        lines = fd.readlines()
    gene_num_list = []
    for line in lines:
        items = line.split('\t')
        descriptions = items[2:]
        gene_num_list.append(len(descriptions))

    count = 0
    for num in gene_num_list:
        if num > 50:
            count += 1
    print(1 - count/len(gene_num_list))
    print(max(gene_num_list))
    print(mean(gene_num_list))
    print(median(gene_num_list))

    n, bins, _ = plt.hist(gene_num_list, bins=range(51),
                          color='c', edgecolor='#FFFFFF')
    print(n, bins)
    max_i = 0
    for i in range(len(n) - 1, -1, -1):
        if n[i] > 0:
            max_i = i
            break
    print(bins[max_i])
    print(n[max_i])

    def bias(k):
        if k < 10:
            return 0.4
        elif k < 1000:
            return 0.2
        elif k < 10000:
            return 0.1

    for i in range(1, len(n)):
        plt.text(bins[i] + bias(n[i]), n[i] + 20, str(int(n[i])), fontsize=5)
    plt.title('Number of Gene Each Term')
    plt.xticks([i + 0.5 for i in range(1, 51)],
               [str(i) for i in range(1, 51)], fontsize=6)
    plt.xlabel('Number of Gene')
    plt.ylabel('Count')
    plt.show()


def draw_term_gene_length_distribution(filename):
    """Draw distribution for gene length.

    Args:
        filename: 'term_name_def_descriptions_human.txt'
    """
    with open(filename) as fd:
        lines = fd.readlines()
    gene_len_list = []
    for line in lines:
        items = line.split('\t')
        descriptions = items[2:]
        gene_len_list.append(sum([len(description.split())
                                  for description in descriptions]))

    print(len(gene_len_list))
    print(mean(gene_len_list))
    print(median(gene_len_list))
    print(max(gene_len_list))
    print(min(gene_len_list))

    print(len([1 for gene_len in gene_len_list if gene_len < 100])
          / len(gene_len_list))
    print(len([1 for gene_len in gene_len_list if gene_len < 200])
          / len(gene_len_list))
    print(len([1 for gene_len in gene_len_list if gene_len < 500])
          / len(gene_len_list))

    n, bins, _ = plt.hist(gene_len_list, bins=range(0, 100, 1),
                          color='m', edgecolor='#FFFFFF')
    print(n, bins)
    max_i = 0
    for i in range(len(n) - 1, -1, -1):
        if n[i] > 0:
            max_i = i
            break
    print(bins[max_i])
    print(n[max_i])

    def bias(k):
        if k < 100:
            return 0.2
        elif k < 1000:
            return 0
        elif k < 10000:
            return -0.05

    for i in range(1, len(n)):
        plt.text(bins[i] + bias(n[i]), n[i] + 5, str(int(n[i])), fontsize=4)
    plt.title('Number of Gene Words Each Term (Less Than 100)')
    plt.xticks([i + 0.5 for i in range(1, 101, 5)],
               [str(i) for i in range(1, 101, 5)], fontsize=4)
    plt.xlabel('Number of Gene Words Each Term')
    plt.ylabel('Count')
    plt.show()


def draw_word_frequency_distribution(filename):
    """Draw distribution for given word frequency.

    Args:
        filename: 'all_word_frequency.txt'
    """
    with open(filename) as fd:
        lines = fd.readlines()
    frequency_list = []
    for line in lines:
        word, frequency = line.split()
        if 1 < int(frequency) < 40:
            frequency_list.append(int(frequency))
    plt.hist(frequency_list, bins=40)
    plt.show()


def show_words_not_in_vocabulary(target_file, vocab_file):
    """Calculate the number of words not in vocabulary.

    Args:
        target_file: 'go_dict_clean_lower.txt'
        vocab_file: 'glove.6B.50d.frequency_more_than_3.txt'
    """
    vocab_set = set()
    with open(vocab_file) as fd:
        lines = fd.readlines()
        for line in lines:
            vocab_set.add(line.split()[0])
    with open(target_file) as fd:
        lines = fd.readlines()
        count = 0
        for line in lines:
            term_name = line.split()[1]
            for word in term_name.split():
                if word not in vocab_set:
                    count += 1
                    break
        print(count, len(lines), count / len(lines))
