<<<<<<< HEAD
# coding=UTF-8
=======
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
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
<<<<<<< HEAD
    s = re.sub("\n", " ", s)

    # s = re.sub(r" (\d+) "," <NUM> ", s)
=======
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
    return s

def filterPair(p):
    return len(p[0]) < MAX_LENGTH and \
        len(p[1]) < MAX_LENGTH and \
        len(p[2]) < MAX_LENGTH

<<<<<<< HEAD
def filterPairs(pairs,method):
    if method == "concate":
        return [pair for pair in pairs if filterPair(pair)]
    else:
        return [pair for pair in pairs if filterPair([pair[0],pair[1]," ".join(pair[2])])]
=======
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530

def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


<<<<<<< HEAD
def prepareData(abs_file_path,method):
=======
def prepareData(abs_file_path):
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
    '''
    :param abs_file_path: 运行的绝对路径
    :return: 本体的名字，描述以及基因组描述
    '''
<<<<<<< HEAD
    file = open(abs_file_path + "/data/data_clean_lower.txt", "r")
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
    return filterPairs(pairs,method)

def prepareGlove(voc,abs_file_path):

    file2 = open(abs_file_path +"/"+ "data/vocab.txt", "r")
    line2 = file2.readline()
    words2 = []
    while(line2):
        part = line2.split(" ")
        word = part[0]
        words2.append(word)
        line2 = file2.readline()

    for i in range(3,voc.num_words):
        word = voc.index2word[i]
        if word not in words2 :
            print("not in",ord(voc.index2word[i]))




=======
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
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530

def indexesFromSentence(voc, sentence):
    '''
    :param voc: 词汇库
    :param sentence: 句子
    :return: 句子中每个字符的编码序号
    '''
<<<<<<< HEAD
    indexs = []
    for word in sentence.split(" "):
        if word in voc.word2index:
            indexs.append(voc.word2index[word])
    return indexs


def tensorFromSentence(voc,word2glove, sentence,method,device):
=======
    return [voc.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(voc, sentence):
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
    '''
    :param voc:词汇库
    :param sentence: 句子
    :return: 根据句子生成的tensor
    '''
<<<<<<< HEAD
    if method == "onehot":
        indexes = indexesFromSentence(voc, sentence)
        indexes.append(EOS_token)
        result = torch.tensor(indexes, dtype = torch.long).view(-1, 1)
    if method == "glove":

        tensors = gloveFromSentence(word2glove,sentence)
        tensor_size = tensors[0].size(0)
        tensors.append(torch.zeros(tensor_size, dtype = torch.float,device = device))
        result = tensors
    return result

def tensorsFromPair(voc, word2glove,input,target,method,device):
=======
    indexes = indexesFromSentence(voc, sentence)
    indexes.append(EOS_token)
    result = torch.tensor(indexes, dtype = torch.long).view(-1, 1)

    return result

def tensorsFromPair(voc,input,target):
>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
    '''
    :param voc: 词汇库
    :param input: 源文本
    :param target: 目标文本
    :return: 两者的tensor元组
    '''
<<<<<<< HEAD
    input_tensor = tensorFromSentence(voc,word2glove, input, method = method, device = device)
    target_tensor = tensorFromSentence(voc,word2glove, target, method = "onehot",device = device)
    # print(input_tensor)
    return (input_tensor, target_tensor)

def gloveFromSentence(word2glove,sentence):
    split = sentence.split(" ")
    gloves = []
    for word in split:
        if word in word2glove:
            gloves.append(torch.tensor(word2glove[word],dtype = torch.float,device = device))
    return gloves

# def word2glove(word):
#     try :
#         return
#     except KeyError:
#         return
=======
    input_tensor = tensorFromSentence(voc, input)
    target_tensor = tensorFromSentence(voc, target)
    return (input_tensor, target_tensor)

>>>>>>> 88cd809551d2f7302bee3f26c8eeccf66e061530
