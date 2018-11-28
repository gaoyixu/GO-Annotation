# coding=UTF-8
import re
from util.datautil import *
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS"
        }
        self.num_words = 3

    def initVoc(self,path,method):
        voc_file = open(path, "r")
        line = voc_file.readline()
        if method == "onehot":
            while (line):

                line = normalizeString(str(line))
                self.addSentence(line)
                line = voc_file.readline()
            voc_file.close()
            return
        else:
            word2glove = {}
            while(line):
                part = line[:-1].split(" ")
                word = part[0]
                self.addWord(word)
                glove = part[1:]
                glove = [float(x) for x in glove]
                word2glove[word] = glove
                line = voc_file.readline()
            self.num_words += len(word2glove)
            voc_file.close()
            return word2glove



    def addSentence(self, sentence):
        for word in re.split("[ \t]", sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print("keep_words {}/{} - {:4f}".format(
            len(keep_words), len(self.word2index),
            len(keep_words) / len(self.word2index)))
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS"
        }
        self.num_words = 3
        for word in keep_words:
            self.addWord(word)