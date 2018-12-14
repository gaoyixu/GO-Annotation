# coding=UTF-8
from __future__ import print_function
from Voc import Voc
from util.evalutil import *
from util.datautil import *
from decoder import DecoderRNN
from encoder import CombineEncoderRNN
from encoder import EncoderRNN
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

hidden_size = 512

encoder_save_path = "model/combineEncoder3.pth"
decoder_save_path = "model/combineDecoder3.pth"
combiner_save_path = "model/combineCombiner3.pth"
model1 = torch.load(current_dir + "/" + encoder_save_path)
model2 = torch.load(current_dir + "/" + decoder_save_path)
model3 = torch.load(current_dir + "/" + combiner_save_path)
combineEvaluateTotally(model1.to(device), model2.to(device),model3.to(device),voc,testpairs,len(testpairs))