import numpy as np
from matplotlib import pyplot as plt
from typing import Dict
import math
import json
import operator
import pandas as pd
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
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

if True:
    for word in allwords:
        IDF[word] = math.log(len(words) / len([1 for x in words if word in x]))
    IDFjson = json.dumps(IDF)
    idffile = open("IDF_data_clean_lower.data", "w")
    idffile.write(IDFjson)
else:
    idffile = open("IDF_data_clean_lower.data", "r")
    IDF = json.loads(idffile.read())

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

model = LogisticRegression(C=10)
model.fit(X_train, Y_train)
print(1 - sum(Y_train) / len(Y_train))
print(1 - sum(Y_test) / len(Y_test))
print(model.score(X_train, Y_train))
print(model.score(X_test, Y_test))

# for i in range(29):
#     AppearDict[i] = []

# postfidf = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# negtfidf = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# for i in range(len(content)):
#     positiveweight = 0.0
#     negativeweight = 0.0
#     wordtfidf = dict()
#     for word in wordsofname[i]:
#         idf = IDF[word]
#         tf = words[i].count(word) / len(words[i])
#         tfidf = tf * idf
#         if word not in stopWords:
#             wordtfidf[word] = tfidf
#             if word in wordsofdescription[i]:
#                 positiveweight += tfidf
#             else:
#                 negativeweight += tfidf
#     sordedwordbytfidf = list(reversed(sorted(wordtfidf.items(), key=operator.itemgetter(1))))
#     if i < 10:
#         print(sordedwordbytfidf)
#     for j, v in enumerate(sordedwordbytfidf):
#         if j >= 10:
#             break
#         if v[0] in wordsofdescription[i]:
#             postfidf[j] += 1
#         else:
#             negtfidf[j] += 1
#     temp = len(wordsofname[i])
#     AppearDict[temp].append(positiveweight / (positiveweight + negativeweight))
