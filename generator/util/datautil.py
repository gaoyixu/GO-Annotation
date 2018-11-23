import re
import torch
import unicodedata
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
    return len(p[0]) < MAX_LENGTH and \
        len(p[1]) < MAX_LENGTH and \
        len(p[2]) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def prepareData(abs_file_path):
    '''
    :param abs_file_path: 运行的绝对路径
    :return: 本体的名字，描述以及基因组描述
    '''
    file = open(abs_file_path + "\data\data_clean.txt", "r")
    line = file.readline()
    pairs = []
    while (line):
        part = line.split("\t")
        otology_name = normalizeString(part[0])
        otology_descri = normalizeString(part[1])
        genes = [normalizeString(s) for s in part[2:]]
        origin_genes = " ".join(genes)
        pairs.append((otology_name, otology_descri, origin_genes))
        line = file.readline()
    return filterPairs(pairs)

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
    result = torch.tensor(indexes, dtype = torch.long).view(-1, 1)

    return result

def tensorsFromPair(voc,input,target):
    '''
    :param voc: 词汇库
    :param input: 源文本
    :param target: 目标文本
    :return: 两者的tensor元组
    '''
    input_tensor = tensorFromSentence(voc, input)
    target_tensor = tensorFromSentence(voc, target)
    return (input_tensor, target_tensor)

