import numpy as np
from matplotlib import pyplot as plt
from typing import Dict
import math
import json

AppearDict = dict()

file = open("./term_name_def_descriptions.txt")
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

for i in range(len(content)):
    positiveweight = 0.0
    negativeweight = 0.0
    for word in wordsofname[i]:
        idf = IDF[word]
        tf = words[i].count(word) / len(words[i])
        tfidf = tf * idf
        if word in wordsofdescription[i]:
            positiveweight += tfidf
        else:
            negativeweight += tfidf
    temp = len(wordsofname[i])
    AppearDict[temp].append(positiveweight / (positiveweight + negativeweight))

def mean(a):
    if len(a) == 0:
        return 0
    return sum(a) / len(a)

for i in range(29):
    AppearDict[i] = mean(AppearDict[i])

for key in sorted(AppearDict.keys()):
    print(key, sum(AppearDict[key])/len(AppearDict[key]))

plt.bar(range(1, 29), [AppearDict[i] for i in range(1, 29)], color="blue")
plt.xlabel("Number of words in name")
plt.ylabel("frequency of word in name appearing in description")
plt.show()
