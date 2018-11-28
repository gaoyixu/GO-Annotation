# coding=UTF-8
from __future__ import print_function
from Voc import Voc
from util.evalutil import *
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)

voc_path = abs_file_path + "/data/data_glove.vectors.300d.txt"
print(voc_path)
voc = Voc("total")
word2glove = voc.initVoc(voc_path,"glove")

pairs = prepareData(abs_file_path,"concate")
print(len(pairs))
encoder_save_path = "encoder3.pth"
decoder_save_path = "decoder3.pth"
model1 = torch.load(current_dir+"/"+encoder_save_path)
model2 = torch.load(current_dir+"/"+decoder_save_path)
evaluateRandomly(voc,word2glove,pairs,model1.to(device) ,model2.to(device),n = 10)