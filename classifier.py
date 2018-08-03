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
IDF: Dict[str, float] = dict()

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

# words_not_in_description = set()
# frequency_of_words_not_in_description = []

# for i in range(len(content)):
#     for word in set(wordsofname[i]):
#         if word in stopWords:
#             continue
#         if word not in wordsofdescription[i]:
#             words_not_in_description.add(word)

# for word in words_not_in_description:
#     frequency_of_words_not_in_description.append(sum(
#         [term.count(word) for term in wordsofdescription]))

# allwords_in_name = {x for sublist in wordsofname for x in sublist}
# print(frequency_of_words_not_in_description.count(0) / len(allwords_in_name))

# plt.figure()
# temp = [x if x < 10 else 10 for x in frequency_of_words_not_in_description]
# plt.hist(temp)
# plt.savefig("frequency_of_words_not_in_description")



X = []
Y = []

for i in range(len(content)):
    description_seperated = content[i][1:]
    for word in set(wordsofdescription[i]):
        if word in stopWords:
            continue
        idf = IDF[word]
        tf = wordsofdescription[i].count(word) / len(wordsofdescription[i])
        tfidf = idf * tf
        wordlength = len(word)
        idf_local = math.log(len(description_seperated) /
                        sum([word in des.split(" ") for des in description_seperated]))
        X.append([tf, idf, tfidf, wordlength, idf_local])
        Y.append(float(word in wordsofname[i]))


def DuplicatePositiveSample(X, Y, positive_score):
    positive_number = sum([y > 0.5 for y in Y])
    negative_number = len(Y) - positive_number
    if positive_number / len(Y) > positive_score:
        return X, Y
    actual_number = int(negative_number / (1 - positive_score)) + 1
    X_return = np.zeros([actual_number, X.shape[1]])
    Y_return = np.zeros([actual_number])
    X_return[:len(Y), :] = X
    Y_return[:len(Y)] = Y
    positive_sample = [i for i in range(len(Y)) if Y[i] > 0.5]
    for i in range(len(Y), actual_number):
        j = random.choice(positive_sample)
        X_return[i, :] = X[j, :]
        Y_return[i] = Y[j]
    return X_return, Y_return

X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

X, Y = DuplicatePositiveSample(X, Y, 0.5)

meanX = X.mean(0)
varX = X.var(0)
for i in range(X.shape[1]):
    X[:, i] = (X[:, i] -  meanX[i]) / math.sqrt(varX[i])

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model = LogisticRegression(C=10000)
# model = SVC()
# model.fit(X_train, Y_train)
model.fit(X, Y)

# probtest = model.predict_proba(X_test)

top_one_accuracy = []
top_two_accuracy = []
top_three_accuracy = []

for i in range(len(content)):
    word_candidate = [word for word in set(
        wordsofdescription[i]) if word not in stopWords]
    description_seperated = content[i][1:]
    def get_prior(word, i, description_seperated):
        idf = IDF[word]
        tf = wordsofdescription[i].count(word) / len(wordsofdescription[i])
        tfidf = idf * tf
        wordlength = len(word)
        idf_local = math.log(len(description_seperated) /
                        sum([word in des.split(" ") for des in description_seperated]))
        feature = np.array([tf, idf, tfidf, wordlength, idf_local])
        feature = (feature - meanX) / np.sqrt(varX)
        return model.predict_proba(feature.reshape(1, -1))[0, 1]
    word_candidate = sorted(word_candidate, key=lambda word: -get_prior(word, i, description_seperated))
    top_one = word_candidate[:1]
    top_two = word_candidate[:2]
    top_three = word_candidate[:3]
    top_one_accuracy.append(sum([float(word in wordsofname[i])
                                 for word in top_one]) / len(top_one))
    top_two_accuracy.append(sum([float(word in wordsofname[i])
                            for word in top_two]) / len(top_two))
    top_three_accuracy.append(sum([float(word in wordsofname[i]) for word in top_three]) / len(top_three))

print([mean(top_one_accuracy), mean(top_two_accuracy), mean(top_three_accuracy)])


# FNs = []
# FPs = []

# for threshhold in np.linspace(0, 1, 1001):
#     Y_predict = 1.0 * (probtest > threshhold)
#     Y_predict = Y_predict[:, 1]
#     FN = ((1 - Y_predict) * (Y_test)).sum() / (Y_test).sum()
#     FP = (Y_predict * (1 - Y_test)).sum() / (1 - Y_test).sum()
#     FNs.append(FN)
#     FPs.append(FP)

# FNn = np.array(FNs)
# FPn: ndarray = np.array(FPs)

# plt.figure()
# plt.scatter(1 - FNn, 1 - FPn)
# plt.xlabel("1 - False Negative")
# plt.ylabel("1 - False Positive")
# plt.savefig("FP-FN-SVM.png", dpi=300)
