import re
import torch
import unicodedata
import random
from collections import defaultdict
MAX_LENGTH = 397
import os
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor_word_frequency = defaultdict(int)

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
    # used for split the line as we define
    testIndex = []
    split_line = split_file.readline()
    index = 0
    while(split_line):
        if int(split_line[:-1]) == 0:
            testIndex.append(index)
        index += 1
        split_line = split_file.readline()
    print("testLen",len(testIndex))

    # get the pairs for train and test
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
        genes = list(set(genes))

        # insert the frequency count
        # for s in genes:
        #     indexes = [voc.word2index[word] for word in s.split(" ")]
        #     indexes.append(EOS_token)
        #     tensor = torch.tensor(indexes, dtype=torch.long).view(-1, 1)
        #     for i in tensor.size(0):
        #         tensor_word_frequency[i] += 1



        if method == "concate":
            origin_genes = " ".join(genes)
            pairs.append((otology_name, otology_descri, origin_genes))
        else:
            origin_genes = genes
            pairs.append((otology_name, otology_descri, origin_genes))
        line = file.readline()
    file.close()
    processedPairs = filterPairs(pairs,method)
    # print("set",set(list(range(len(processedPairs))))-set(testIndex))
    trainIndex = list(set(list(range(len(processedPairs))))-set(testIndex))
    print("trainLen", len(trainIndex))
    print()
    trainpairs = [
        processedPairs[i] for i in trainIndex
    ]
    testPairs = [
        processedPairs[i] for i in testIndex
    ]
    # print(testIndex)
    return trainpairs,testPairs


def indexesFromSentence(voc, sentence):
    '''
    :param voc: 词汇库
    :param sentence: 句子
    :return: 句子中每个字符的编码序号
    '''
    # print(voc.word2index)
    return [voc.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(voc, sentence):
    '''
    :param voc:词汇库
    :param sentence: 句子
    :return: 根据句子生成的tensor
    '''

    indexes = indexesFromSentence(voc, sentence)
    indexes.append(EOS_token)
    result = torch.tensor(indexes, dtype=torch.long).view(-1, 1).type(torch.cuda.LongTensor)
    # print(result)

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
        random.shuffle(input_tensor)
        if len(input_tensor) > MAX_LENGTH:
            random.sample(input_tensor,MAX_LENGTH)
        target_tensor = tensorFromSentence(voc, target)
    elif model == "random":
        input_tensor = [tensorFromSentence(voc, input[random.randint(0,len(input)-1)])]
        target_tensor = tensorFromSentence(voc, target)
    else:
        input_tensor = tensorFromSentence(voc, input)
        target_tensor = tensorFromSentence(voc, target)
    return [input_tensor, target_tensor]

def initadj(voc,pairs,current_dir):
    adj = defaultdict(int)
    if os.path.exists(current_dir+"/adj.txt"):
        X = readadj(current_dir + "/adj.txt")
    else:
        for pair in pairs:
            name = pair[0]
            for gene in pair[2]:
                for geneword in gene.split(" "):
                    for nameword in name:
                        if geneword in voc.word2index and nameword in voc.word2index:
                            # print(geneword,nameword)
                            adj[(voc.word2index[geneword],voc.word2index[nameword])] -= 1
                            adj[(voc.word2index[nameword], voc.word2index[geneword])] -= 1
                            adj[(voc.word2index[geneword], voc.word2index[geneword])] += 1
                            adj[(voc.word2index[nameword], voc.word2index[nameword])] += 1
        # for index in range(voc.num_words):
        #     adj[index,index] = 1
        with open(current_dir+"/adj.txt","w") as file:
            for index1 in range(voc.num_words):
                adjw = [adj[(index1,index2)] for index2 in range(voc.num_words)]
                # print(" ".join([str() for x in adjw]))
                file.write(" ".join([str(x) for x in adjw])+"\n")
        X = readadj(current_dir + "/adj.txt")
    print("get the adj")
    return X
def readadj(file_name):
    # inputMatrix = np.zeros((34,34))
    with open(file_name) as file:
        matrix = []
        for line in file:
            array = []
            line = line.split(" ")
            if line[len(line)-1].endswith("\n"):
                line[len(line)-1] = line[len(line)-1][:-1]
            for vertex in line:
                array.append(float(vertex))
            matrix.append(array)
    # print("origin",matrix)
    return matrix

