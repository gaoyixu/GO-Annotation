# coding=UTF-8
from __future__ import print_function
import random
import os
from util.datautil import *
from nltk.translate.bleu_score import sentence_bleu
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)
MAX_LENGTH = 2000
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#GPU form
def evaluate(voc,word2glove,encoder, decoder, sentence, max_length = MAX_LENGTH):
    '''
    :param encoder: 编码器
    :param decoder: 解码器
    :param sentence: 输入的文本
    :param max_length:
    :return:
    '''
    with torch.no_grad():
        input_tensors = tensorFromSentence(voc,word2glove, sentence, "glove",device = torch.device("cuda"))
        input_length = len(input_tensors)
        encoder_hidden = encoder.initHidden(device = torch.device("cuda"))

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size,device = torch.device("cuda") )
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensors[ei],
                                                     encoder_hidden)
            encoder_hidden = encoder_hidden
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([EOS_token]).type(torch.cuda.LongTensor)
        decoder_hidden = encoder_hidden.type(torch.cuda.FloatTensor)
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output1, decoder_hidden1, decoder_attention1 = decoder(
                decoder_input, decoder_hidden, encoder_outputs,"cuda"
            )
            decoder_output, decoder_hidden, decoder_attention = decoder_output1.type(torch.cuda.LongTensor), \
                                                                decoder_hidden1.type(torch.cuda.FloatTensor), decoder_attention1
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(voc.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(voc,word2glove,pairs,encoder, decoder, n = 10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[2])
        print('=', pair[0])
        output_words, attentions = evaluate(voc, word2glove, encoder, decoder, pair[2])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def bleu(targets,outputs):
    reference = [targets]
    candidate = outputs
    score = sentence_bleu(reference, candidate)
    return score
