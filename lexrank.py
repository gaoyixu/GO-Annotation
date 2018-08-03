import numpy as np
from numpy import ndarray
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
from typing import Dict
import math
import json
import operator
import pandas as pd
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import random
stopWords = set(stopwords.words('english'))

def mean(array):
    if not array:
        return 0
    return sum(array) / len(array)

AppearDict = dict()

file = open("./data/data_clean_lower.txt")
content = file.readlines()
IDF = dict()

content = [line[:-1].split("\t") for line in content]
name = [x[0] for x in content]
description = [" ".join(x[1:]) for x in content]
wordsofname = [x.split(" ") for x in name]
wordsofname = [[y for y in x if y != ""] for x in wordsofname]
wordsofdescription = [x.split(" ") for x in description]
wordsofdescription = [[y for y in x if y != ""] for x in wordsofdescription]
words = [x + y for x, y in zip(wordsofname, wordsofdescription)]

allwords = {x for sublist in words for x in sublist}

if False:
    for word in allwords:
        IDF[word] = math.log(len(words) / len([1 for x in words if word in x]))
    IDFjson = json.dumps(IDF)
    idffile = open("IDF_data_clean_lower.data", "w")
    idffile.write(IDFjson)
else:
    idffile = open("IDF_data_clean_lower.data", "r")
    IDF = json.loads(idffile.read())

def idf_modified_dot(tf_hash_x, tf_hash_y=None):
    inner_product = 0
    if tf_hash_y is None:
        for value in tf_hash_x.values():
            inner_product += value ** 2
        return inner_product
    for word in set(tf_hash_x.keys()).intersection(set(tf_hash_y.keys())):
        inner_product += tf_hash_x[word] * tf_hash_y[word]
    return inner_product

def idf_modified_cos(tf_hash_x, tf_hash_y):
    return idf_modified_dot(tf_hash_x, tf_hash_y) / math.sqrt(idf_modified_dot(tf_hash_x) * idf_modified_dot(tf_hash_y))

Lexrank = []
tfidf_score_all = []

def update(word_dict, word, value):
    if word in word_dict:
        word_dict[word] += value
    else:
        word_dict[word] = value

for i in range(len(content)):
    tfidf_Hash = []
    description_seperated = content[i][1:]
    description_seperated = [desc for desc in description_seperated if desc]
    n = len(description_seperated)
    for j in range(n):
        temp_hash = dict()
        description_seperated_j = description_seperated[j].split(" ")
        for word in set(description_seperated_j):
            if not word:
                continue
            if word in stopWords:
                continue
            if not word[0].isalpha:
                continue
            idf = IDF[word]
            tf = description_seperated_j.count(word) / len(description_seperated_j)
            temp_hash[word] = idf * tf
        if not temp_hash:
            print("Fault")
        tfidf_Hash.append(temp_hash)
    M = np.zeros([n, n])
    for x in range(n):
        for y in range(n):
            M[x, y] = idf_modified_cos(tfidf_Hash[x], tfidf_Hash[y])
    for x in range(n):
        M[x, :] /= M[x, :].sum()
    M_modified = 0.15 * np.eye(n) + 0.85 * M
    p = np.zeros([n, 1]) + 1 / n
    while True:
        p_new = np.dot(M_modified.T, p)
        if np.linalg.norm(p_new - p) < 1e-5:
            break
        p = p_new
    Lexrank.append(p)
    tfidf_score = dict()
    for j in range(n):
        description_seperated_j = description_seperated[j].split(" ")
        for word in set(description_seperated_j):
            if not word:
                continue
            if word in stopWords:
                continue
            if not word[0].isalpha:
                continue
            idf = IDF[word]
            tf = description_seperated_j.count(word) / len(description_seperated_j)
            update(tfidf_score, word, idf * tf * float(p[j]))
    tfidf_score_all.append(tfidf_score)

top_one_accuracy = []
top_two_accuracy = []
top_three_accuracy = []

for i in range(len(content)):
    word_candidate = [word for word in set(
        wordsofdescription[i]) if word not in stopWords and word and word[0].isalpha()]
    description_seperated = content[i][1:]
    def get_prior(word, i):
        return tfidf_score_all[i][word]
    word_candidate = sorted(word_candidate, key=lambda word: -get_prior(word, i))
    top_one = word_candidate[:1]
    top_two = word_candidate[:2]
    top_three = word_candidate[:3]
    top_one_accuracy.append(sum([float(word in wordsofname[i])
                                 for word in top_one]) / len(top_one))
    top_two_accuracy.append(sum([float(word in wordsofname[i])
                            for word in top_two]) / len(top_two))
    top_three_accuracy.append(sum([float(word in wordsofname[i]) for word in top_three]) / len(top_three))

print([mean(top_one_accuracy), mean(top_two_accuracy), mean(top_three_accuracy)])

# def DuplicatePositiveSample(X, Y, positive_score):
#     positive_number = sum([y > 0.5 for y in Y])
#     negative_number = len(Y) - positive_number
#     if positive_number / len(Y) > positive_score:
#         return X, Y
#     actual_number = int(negative_number / (1 - positive_score)) + 1
#     X_return = np.zeros([actual_number, X.shape[1]])
#     Y_return = np.zeros([actual_number])
#     X_return[:len(Y), :] = X
#     Y_return[:len(Y)] = Y
#     positive_sample = [i for i in range(len(Y)) if Y[i] > 0.5]
#     for i in range(len(Y), actual_number):
#         j = random.choice(positive_sample)
#         X_return[i, :] = X[j, :]
#         Y_return[i] = Y[j]
#     return X_return, Y_return

# X = np.array(X, dtype=float)
# Y = np.array(Y, dtype=float)

# X, Y = DuplicatePositiveSample(X, Y, 0.5)

# meanX = X.mean(0)
# varX = X.var(0)
# for i in range(X.shape[1]):
#     X[:, i] = (X[:, i] -  meanX[i]) / math.sqrt(varX[i])

# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# model = LogisticRegression(C=10000)
# # model = SVC()
# # model.fit(X_train, Y_train)
# model.fit(X, Y)

# # probtest = model.predict_proba(X_test)



# # FNs = []
# # FPs = []

# # for threshhold in np.linspace(0, 1, 1001):
# #     Y_predict = 1.0 * (probtest > threshhold)
# #     Y_predict = Y_predict[:, 1]
# #     FN = ((1 - Y_predict) * (Y_test)).sum() / (Y_test).sum()
# #     FP = (Y_predict * (1 - Y_test)).sum() / (1 - Y_test).sum()
# #     FNs.append(FN)
# #     FPs.append(FP)

# # FNn = np.array(FNs)
# # FPn: ndarray = np.array(FPs)

# # plt.figure()
# # plt.scatter(1 - FNn, 1 - FPn)
# # plt.xlabel("1 - False Negative")
# # plt.ylabel("1 - False Positive")
# # plt.savefig("FP-FN-SVM.png", dpi=300)
