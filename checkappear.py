import numpy as np
from matplotlib import pyplot as plt
from typing import Dict
import math
import json
import operator
import pandas as pd
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

AppearDict = dict()

file = open("./term_name_def_descriptions_from_gene_info_and_gene2go.txt")
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
    idffile = open("IDF.data", "w")
    idffile.write(IDFjson)
else:
    idffile = open("IDF.data", "r")
    IDF = json.loads(idffile.read())

for i in range(29):
    AppearDict[i] = []

postfidf = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
negtfidf = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(content)):
    positiveweight = 0.0
    negativeweight = 0.0
    wordtfidf = dict()
    for word in wordsofname[i]:
        idf = IDF[word]
        tf = words[i].count(word) / len(words[i])
        tfidf = tf * idf
        if word not in stopWords:
            wordtfidf[word] = tfidf
            if word in wordsofdescription[i]:
                positiveweight += tfidf
            else:
                negativeweight += tfidf
    sordedwordbytfidf = list(reversed(sorted(wordtfidf.items(), key=operator.itemgetter(1))))
    if i < 10:
        print(sordedwordbytfidf)
    for j, v in enumerate(sordedwordbytfidf):
        if j >= 10:
            break
        if v[0] in wordsofdescription[i]:
            postfidf[j] += 1
        else:
            negtfidf[j] += 1
    temp = len(wordsofname[i])
    AppearDict[temp].append(positiveweight / (positiveweight + negativeweight))

ax = plt.bar(range(10), [postfidf[i] / (negtfidf[i] + postfidf[i])
                        for i in range(10)], width=0.8, color="blue")
plt.xlabel("tfidf rank")
plt.xticks(range(10), [str(i + 1) for i in range(0, 10)])
plt.ylabel("fraction of word appearing in description")
plt.ylim([0, 1])

rects = ax.patches

labels = [str(postfidf[i]/(negtfidf[i] + postfidf[i]))[:4] for i in range(10)]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')

plt.savefig("tfidfrank.png", dpi=300)

def mean(a):
    if not a:
        return 0
    return sum(a) / len(a)

for i in range(29):
    AppearDict[i] = mean(AppearDict[i])

plt.figure()
plt.bar(range(1, 29), [AppearDict[i] for i in range(1, 29)], color="blue")
plt.xlabel("Number of words in name")
plt.ylabel("frequency of word in name appearing in description")
plt.savefig("frequencyofappearing.png", dpi=300)
