import numpy as np
from numpy import ndarray
import matplotlib
matplotlib.use('agg')
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
stopWords = set(stopwords.words('english'))

AppearDict = dict()

file = open("./data_clean_lower.txt")
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
    for word in set(wordsofdescription[i]):
        if word in stopWords:
            continue
        idf = IDF[word]
        tf = wordsofdescription[i].count(word) / len(wordsofdescription)
        tfidf = idf * tf
        wordlength = len(word)
        X.append([tf, idf, tfidf, wordlength])
        Y.append(float(word in wordsofname[i]))

X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# model = LogisticRegression(C=10)
model = SVC()
model.fit(X_train, Y_train)

probtest = model.predict_proba(X_test)

FNs = []
FPs = []

for threshhold in np.linspace(0, 1, 1001):
    Y_predict = 1.0 * (probtest > threshhold)
    Y_predict = Y_predict[:, 1]
    FN = ((1 - Y_predict) * (Y_test)).sum() / (Y_test).sum()
    FP = (Y_predict * (1 - Y_test)).sum() / (1 - Y_test).sum()
    FNs.append(FN)
    FPs.append(FP)

FNn = np.array(FNs)
FPn: ndarray = np.array(FPs)

plt.figure()
plt.scatter(1 - FNn, 1 - FPn)
plt.xlabel("1 - False Negative")
plt.ylabel("1 - False Positive")
plt.savefig("FP-FN-SVM.png", dpi=300)
