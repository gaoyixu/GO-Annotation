import re
import torch
import unicodedata
import random
MAX_LENGTH = 2000
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EOS_token = 1


def normalizeString(s):
    '''
    :param s:字符串
    :return: 小写化，去特殊符号的字符串
    '''
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPair(p):
    # return len(p[0]) < MAX_LENGTH and \
    #     len(p[1]) < MAX_LENGTH and \
    #     len(p[2]) < MAX_LENGTH
    return True


def filterPairs(pairs,method):
    if method == "concate":
        return [pair for pair in pairs if filterPair(pair)]
    else:
        return [pair for pair in pairs if filterPair([pair[0],pair[1]," ".join(pair[2])])]


def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def prepareData(abs_file_path, method):
    '''
    :param abs_file_path: 运行的绝对路径
    :return: 本体的名字，描述以及基因组描述
    '''
    file = open(abs_file_path + "/data/data_clean_lower.txt", "r")
    split_file = open(abs_file_path + "/data/train_test_label.txt")
    testIndex = []
    split_line = split_file.readline()
    index = 0
    while(split_line):
        if int(split_line[:-1]) == 0:
            testIndex.append(index)
        split_line = split_file.readline()
    print("testLen",len(testIndex))


    line = file.readline()
    pairs = []
    while (line):

        part = line.split("\t")
        for word in part:
            if len(word)<1:
                part = part.remove(word)
        otology_name = normalizeString(part[0])
        otology_descri = normalizeString(part[1])
        genes = [normalizeString(s) for s in part[2:]]

        if method == "concate":
            origin_genes = " ".join(genes)
            pairs.append((otology_name, otology_descri, origin_genes))
        else:
            origin_genes = genes
            pairs.append((otology_name, otology_descri, origin_genes))
        line = file.readline()
    file.close()
    processedPairs = filterPairs(pairs,method)
    trainIndex = list(set(range(len(processedPairs)))-set(testIndex))
    print("trainLen", len(trainIndex))
    trainpairs = [
        processedPairs[i] for i in trainIndex
    ]
    testPairs = [
        processedPairs[i] for i in testIndex
    ]
    return trainpairs,testPairs


def indexesFromSentence(voc, sentence):
    '''
    :param voc: 词汇库
    :param sentence: 句子
    :return: 句子中每个字符的编码序号
    '''
    return [voc.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(voc, sentence):
    '''
    :param voc:词汇库
    :param sentence: 句子
    :return: 根据句子生成的tensor
    '''
    indexes = indexesFromSentence(voc, sentence)
    indexes.append(EOS_token)
    result = torch.tensor(indexes, dtype=torch.long).view(-1, 1)

    return result


def tensorsFromPair(voc, input, target, model):
    '''
    :param voc: 词汇库
    :param input: 源文本
    :param target: 目标文本
    :return: 两者的tensor元组
    '''
    if model == "context":
        input_tensor = [
            tensorFromSentence(voc, input[i]) for i in range(len(input))
        ]
        target_tensor = tensorFromSentence(voc, target)
    elif model == "random":
        input_tensor = [tensorFromSentence(voc, input[random.randint(0,len(input)-1)])]
        target_tensor = tensorFromSentence(voc, target)
    else:
        input_tensor = tensorFromSentence(voc, input)
        target_tensor = tensorFromSentence(voc, target)
    return [input_tensor, target_tensor]