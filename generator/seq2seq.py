# coding=UTF-8
from __future__ import print_function
import torch.nn as nn
import random
import os
from Voc import Voc
from util.datautil import *
from util.timeutil import *
from util.plotutil import *
from util.evalutil import *
from decoder import AttnDecoderRNN
from encoder import EncoderRNN
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)
MAX_LENGTH = 2000
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.7

def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, max_length = MAX_LENGTH):
    '''
    :param input_tensor: 输入的整个字段的每个word的字符的隐表示集合
    :param target_tensor: 输出的整个字段的每个word的字符的隐表示集合
    :param encoder: 编码器
    :param decoder: 解码器
    :param encoder_optimizer: 编码优化器
    :param decoder_optimizer: 解码优化器
    :param criterion: 评价函数
    :param max_length: 最大的输入输出长度
    :return:
    '''
    encoder_hidden = encoder.initHidden()
    # print("hidden",encoder_hidden.size())
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = len(input_tensor)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0,0]
    decoder_input = torch.tensor([[SOS_token]]).type(torch.cuda.LongTensor)

    decoder_hidden = encoder_hidden

    use_teaching_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teaching_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs ,"cuda"
            )
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, "cuda"
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iter, print_every = 10, plot_every=6000, learning_rate = 0.01):
    '''

    :param encoder: 编码器
    :param decoder: 解码器
    :param n_iter: 迭代次数
    :param print_every: 每次打印时间间隔
    :param plot_every: 每次作图时间间隔
    :param learning_rate: 学习率
    :return:
    '''
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    randomPairs = [random.choice(pairs) for i in range(n_iter)]
    training_pairs = [tensorsFromPair(voc,word2glove,randomPairs[i][2],randomPairs[i][0],"glove",device)  for i in range(n_iter)]
    criterion = nn.NLLLoss()
    for iter in range(1, n_iter + 1):
        training_pair = training_pairs[iter -1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1].type(torch.cuda.LongTensor)
        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iter),
                                         iter, iter / n_iter * 100, print_loss_avg))
    #     if iter % plot_every == 0:
    #         plot_loss_avg = plot_loss_total / plot_every
    #         plot_losses.append(plot_loss_avg)
    #         plot_loss_total = 0
    # showPlot(plot_losses)





voc_path = abs_file_path + "/data/data_glove.vectors.300d.txt"
print(voc_path)
voc = Voc("total")
word2glove = voc.initVoc(voc_path,"glove")

pairs = prepareData(abs_file_path,"concate")
print(len(pairs))

hidden_size = 100


encoder1 = EncoderRNN(voc.num_words, hidden_size).to(device)


attn_decoder1 = AttnDecoderRNN(hidden_size, voc.num_words, dropout = 0.1).to(device)



trainIters(encoder1, attn_decoder1, 1)

encoder_save_path = "encoder3.pth"
decoder_save_path = "decoder3.pth"
torch.save(encoder1, current_dir+'/'+encoder_save_path)
torch.save(attn_decoder1, current_dir+"/"+decoder_save_path)
model1 = torch.load(current_dir+"/"+encoder_save_path)
model2 = torch.load(current_dir+"/"+decoder_save_path)
evaluateRandomly(voc,word2glove,pairs,model1.to(device) ,model2.to(device),n = 10)
