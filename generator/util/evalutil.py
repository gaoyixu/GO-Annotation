# coding=UTF-8
from __future__ import print_function
import random
import os
import warnings
warnings.filterwarnings("ignore")
from util.datautil import *
import torch
EOS_token = 1
from nltk.translate.bleu_score import sentence_bleu
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)
MAX_LENGTH = 397
word_max_length = 30
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
        decoder_input = torch.tensor([SOS_token]).type(torch.cuda.LongTensor)
        decoder_hidden = encoder_hidden.type(torch.cuda.FloatTensor)
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output1, decoder_hidden1, decoder_attention1 = decoder(
                decoder_input, decoder_hidden, encoder_outputs,"cuda"
            )
            decoder_output, decoder_hidden, decoder_attention = decoder_output1.type(torch.cuda.LongTensor), \
                                                                decoder_hidden1.type(torch.cuda.FloatTensor), decoder_attention1
            # data = decoder_attention.data.topk(1)
            # decoder_attentions[di] = data
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
        score = bleu(pair[0],output_words[:-1])
        print('<', output_sentence)
        print("bleu", score)
        print('')

def evaluateTotally(voc,word2glove,pairs,encoder, decoder,length = 1000):
    scoresSum = 0
    for i in range(length):
        pair = pairs[i]
        output_words, attentions = evaluate(voc, word2glove, encoder, decoder, pair[2])
        score = bleu(pair[0], output_words[:-1])
        scoresSum+=score
        if i %100 ==0:
            print("turn",i)
    print("total relu %.4f" % (scoresSum/len(pairs)))
    return

def bleu(targets,outputs):
    outputs = outputs[:-1]
    # print("target", targets[0],len(targets))
    # print("outputs",outputs[0],len(outputs))
    # print("targets",targets)
    # print("output",outputs)
    reference = [targets]
    candidate = outputs
    score = sentence_bleu(reference, candidate)
    return score

def combineEvaluate(encoder, decoder,CombineEncoder,voc, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensors = [
            tensorFromSentence(voc, sentence[i]) for i in range(len(sentence))
        ]
        input_size = len(input_tensors)
        encoder_hidden = encoder.initHidden(device=torch.device("cuda"))
        Combine_hidden = CombineEncoder.initHidden(device=torch.device("cuda"))

        # encoder_outputs = torch.zeros(
        #     max_length, encoder.hidden_size, device=torch.device("cpu"))

        for i in range(input_size):
            input_length = len(input_tensors[i])
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensors[i][ei].type(torch.cuda.LongTensor), encoder_hidden)
                encoder_hidden = encoder_hidden
                # encoder_outputs[ei] += encoder_output[0, 0]
            # if i == 0:
            #     hiddens = encoder_hidden
            # else:
            #     hiddens = hiddens + encoder_hidden
            Combine_output, Combine_hidden = CombineEncoder(
                encoder_hidden.type(torch.cuda.FloatTensor),
                Combine_hidden
            )
        meanhiddens = Combine_hidden

        decoder_input = torch.tensor([SOS_token]).type(torch.cuda.LongTensor)
        decoder_hidden = meanhiddens.type(torch.cuda.FloatTensor)
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output1, decoder_hidden1 = decoder(decoder_input,
                                                       decoder_hidden)
            decoder_output, decoder_hidden = decoder_output1.type(torch.cuda.LongTensor), \
                                                                decoder_hidden1.type(torch.cuda.FloatTensor)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(voc.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words

def AttencombineEvaluate(encoder, decoder,CombineEncoder,voc, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensors = [
            tensorFromSentence(voc, sentence[i]) for i in range(len(sentence))
        ]
        input_size = len(input_tensors)
        encoder_hidden = encoder.initHidden(device=torch.device("cuda"))
        Combine_hidden = CombineEncoder.initHidden(device=torch.device("cuda"))
        Combine_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        # encoder_outputs = torch.zeros(
        #     max_length, encoder.hidden_size, device=torch.device("cpu"))

        for i in range(min(input_size,MAX_LENGTH)):
            input_length = len(input_tensors[i])
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensors[i][ei].type(torch.cuda.LongTensor), encoder_hidden)
                encoder_hidden = encoder_hidden
                # encoder_outputs[ei] += encoder_output[0, 0]
            # if i == 0:
            #     hiddens = encoder_hidden
            # else:
            #     hiddens = hiddens + encoder_hidden
            Combine_output, Combine_hidden = CombineEncoder(
                encoder_hidden.type(torch.cuda.FloatTensor),
                Combine_hidden
            )
            Combine_outputs[i] = Combine_output[0, 0]
        meanhiddens = Combine_hidden

        decoder_input = torch.tensor([SOS_token]).type(torch.cuda.LongTensor)
        decoder_hidden = meanhiddens.type(torch.cuda.FloatTensor)
        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)
        decoder_attentions = []

        for di in range(max_length):
            decoder_output1, decoder_hidden1, decoder_attention = decoder(decoder_input,
                                                       decoder_hidden,Combine_outputs)
            decoder_output, decoder_hidden = decoder_output1.type(torch.cuda.LongTensor), \
                                                                decoder_hidden1.type(torch.cuda.FloatTensor)
            # decoder_attentions[di] = decoder_attention.data
            decoder_attentions.append(decoder_attention.data.topk(1)[1].data)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(voc.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]

def wordAttencombineEvaluate(encoder, decoder,CombineEncoder,voc, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensors = [
            tensorFromSentence(voc, sentence[i]) for i in range(len(sentence))
        ]
        input_size = len(input_tensors)
        encoder_hidden = encoder.initHidden(device=torch.device("cuda"))
        Combine_hidden = CombineEncoder.initHidden(device=torch.device("cuda"))
        Combine_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_outputs = [torch.zeros(word_max_length, encoder.hidden_size, device=device) for i in
                           range(max_length)]
        # encoder_outputs = torch.zeros(
        #     max_length, encoder.hidden_size, device=torch.device("cpu"))
        count = 0
        for i in range(min(input_size, MAX_LENGTH)):
            input_length = len(input_tensors[i])
            for ei in range(min(input_length, word_max_length)):
                encoder_output, encoder_hidden = encoder(
                    input_tensors[i][ei].type(torch.cuda.LongTensor), encoder_hidden)
                encoder_hidden = encoder_hidden
                if count < word_max_length:
                    encoder_outputs[i][count] = encoder_output[0, 0]
                    count += 1
                # encoder_outputs[ei] += encoder_output[0, 0]
            # if i == 0:
            #     hiddens = encoder_hidden
            # else:
            #     hiddens = hiddens + encoder_hidden
            Combine_output, Combine_hidden = CombineEncoder(
                encoder_hidden.type(torch.cuda.FloatTensor),
                Combine_hidden,
                encoder_outputs[i]
            )
            Combine_outputs[i] = Combine_output[0, 0]
        meanhiddens = Combine_hidden

        decoder_input = torch.tensor([SOS_token]).type(torch.cuda.LongTensor)
        decoder_hidden = meanhiddens.type(torch.cuda.FloatTensor)
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output1, decoder_hidden1, decoder_attentions = decoder(decoder_input,
                                                                           decoder_hidden, Combine_outputs)
            decoder_output, decoder_hidden = decoder_output1.type(torch.cuda.LongTensor), \
                                             decoder_hidden1.type(torch.cuda.FloatTensor)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(voc.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]

def combineEvaluateRandomly(encoder, decoder,CombineEncoder,voc,pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(pair)
        print('>', pair[2])
        print('=', pair[0])
        output_words = combineEvaluate(encoder, decoder,CombineEncoder,voc, pair[2])
        score = bleu(pair[0].split(" "), output_words)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print("bleu", score)
        print('')

def combineEvaluateTotally(encoder, decoder,CombineEncoder,voc,pairs,length,model = "normal"):
    scoresSum = 0
    for i in range(length):
        decoder_attentions = None
        pair = pairs[i]
        if model == "attention":
            output_words,decoder_attentions = AttencombineEvaluate(encoder, decoder,CombineEncoder,voc, pair[2])
        elif model == "wordattention":
            output_words,_ = wordAttencombineEvaluate(encoder, decoder,CombineEncoder,voc, pair[2])
        else:
            output_words = combineEvaluate(encoder, decoder, CombineEncoder, voc, pair[2])
        print('>', pair[2])
        print("=",pair[0])
        output_sentence = ' '.join(output_words)
        print("<",output_sentence)
        if model == "attention":
            print("attention",decoder_attentions)
        score = bleu(pair[0].split(" "), output_words[:-1])
        print(score)
        scoresSum += score
        # if i % 20 == 0 and i != 0:
        #     print("current round", i,"/",length,"bleu:","%.4f" % (scoresSum / i))
    print("total relu %.4f" % (scoresSum / length))
    return

if __name__ == "__main__":
    reference = "positive regulation of interferon gamma secretion".split(" ")
    candidate = "positive regulation of <EOS>".split(" ")
    print(bleu(reference,candidate))

