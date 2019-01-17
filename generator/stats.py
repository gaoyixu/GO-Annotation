# For word shared in genes
import json
from Voc import Voc
from util.datautil import *
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)
voc_path = abs_file_path + "/data/data_clean_lower.txt"
voc = Voc("total")
voc.initVoc(voc_path)
trainpairs, testpairs = prepareData(abs_file_path,"context")
trainpairs.extend(testpairs)
print(trainpairs[0][0])
print(trainpairs[0][1])
print(trainpairs[0][2])
descibe = [pair[2] for pair in trainpairs]
bigsentence = "|".join([sentences for i in range(len(trainpairs)) for sentences in descibe[i]])
print(bigsentence)
data = range(1000)
# new_dict = json.dumps({'a': 'Runoob', 'b': 7})

# countdict = {sentences:bigsentence.count(sentences,0,len(bigsentence)) for i in range(len(trainpairs)) for sentences in trainpairs[i][2]}
# print(countdict)
# with open(current_dir + "/record.json", "w") as f:
#     json.dump(countdict, f)
#     print("加载入文件完成...")
with open(current_dir + "/record.json", 'r') as load_f:
    load_dict = json.load(load_f)
    print(type(load_dict))
sortdict = sorted(load_dict,key=load_dict.__getitem__)
counts= defaultdict(int)
for key, value in load_dict.items():
    counts[value] +=1
keys = []
for key, value in counts.items():
    keys.append(key)
# for key in keys:
#     print(key)
keys = sorted(keys)
print("number of genes",len(keys))
plt.hist([counts[key] for key in keys],list(range(1,200,1)), histtype='bar', rwidth=0.8)
plt.xlabel('the number of gene groups the gene in ')
plt.ylabel('the number of genes')

plt.show()