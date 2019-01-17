# coding=UTF-8
from __future__ import print_function
from Voc import Voc
from util.datautil import *
from gcnfile.gcnutil import *
from util.evalutil import *

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)

voc_path = abs_file_path + "/data/data_clean_lower.txt"
voc = Voc("total")
voc.initVoc(voc_path)
trainpairs, testpairs = prepareData(abs_file_path,"context")
descibe = [pair[2] for pair in testpairs]
print("ontology name：",testpairs[0][0],"\nontology token：",testpairs[0][1],"\ngene desciption：",descibe[0],"\n",)
# encodernum = max(map(lambda x:len(x),descibe))
# print(encodernum)
#
# X = initadj(voc,trainpairs,current_dir)
# A , D = preprocess(X)
# print("A",A,"D",D)
#
hidden_size = 256
day = 19
hour = "01"
nowTime = '2018-12-'+str(day)+'-'+str(hour)
encoder_save_path = "model/AttencombineEncoder+" +nowTime+ "hidden"+str(hidden_size)+ "+.pth"
decoder_save_path = "model/AttencombineDecoder+" +nowTime+ "hidden"+str(hidden_size)+ "+.pth"
combiner_save_path = "model/AttencombineCombiner+" +nowTime+ "hidden"+str(hidden_size)+ "+.pth"
model1 = torch.load(current_dir + "/" + encoder_save_path)
model2 = torch.load(current_dir + "/" + decoder_save_path)
model3 = torch.load(current_dir + "/" + combiner_save_path)
combineEvaluateTotally(model1.to(device), model2.to(device),model3.to(device),voc,testpairs,len(testpairs),"attention")