from __future__ import print_function
import torch.nn as nn
import random
import os
from Voc import Voc
from util.datautil import *
from util.timeutil import *
from util.plotutil import *
from decoder import AttnDecoderRNN
from encoder import EncoderRNN
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)
MAX_LENGTH = 2000
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5


def train(input_tensor,
          target_tensor,
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]]).type(torch.cuda.LongTensor)

    decoder_hidden = encoder_hidden

    use_teaching_forcing = True if random.random(
    ) < teacher_forcing_ratio else False

    if use_teaching_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, "cuda")
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, "cuda")
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder,
               decoder,
               n_iter,
               print_every=10,
               plot_every=100,
               learning_rate=0.0001):
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

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    randomPairs = [random.choice(pairs) for i in range(n_iter)]
    training_pairs = [
        tensorsFromPair(voc, randomPairs[i][2], randomPairs[i][0], "plain")
        for i in range(n_iter)
    ]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iter + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0].type(torch.cuda.LongTensor)
        target_tensor = training_pair[1].type(torch.cuda.LongTensor)
        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iter), iter,
                                         iter / n_iter * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensors = tensorFromSentence(voc, sentence)
        input_length = input_tensors.size()[0]
        encoder_hidden = encoder.initHidden(device=torch.device("cpu"))

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=torch.device("cpu"))
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensors[ei],
                                                     encoder_hidden)
            encoder_hidden = encoder_hidden
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([EOS_token]).type(torch.LongTensor)
        decoder_hidden = encoder_hidden.type(torch.FloatTensor)
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output1, decoder_hidden1, decoder_attention1 = decoder(
                decoder_input, decoder_hidden, encoder_outputs, "cpu")
            decoder_output, decoder_hidden, decoder_attention = decoder_output1.type(torch.LongTensor), \
                                                                decoder_hidden1.type(torch.FloatTensor), decoder_attention1
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(voc.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(pair)
        print('>', pair[2])
        print('=', pair[0])
        output_words, attentions = evaluate(encoder, decoder, pair[2])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


voc_path = abs_file_path + "/data/data_clean.txt"
voc = Voc("total")
voc.initVoc(voc_path)
pairs = prepareData(abs_file_path)
print(len(pairs))

hidden_size = 256
encoder1 = EncoderRNN(voc.num_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(
    hidden_size, voc.num_words, dropout=0.1).to(device)
trainIters(encoder1, attn_decoder1, 75000)

encoder_save_path = "encoder3.pth"
decoder_save_path = "decoder3.pth"
torch.save(encoder1, current_dir + '/' + encoder_save_path)
torch.save(attn_decoder1, current_dir + "/" + decoder_save_path)
model1 = torch.load(current_dir + "/" + encoder_save_path)
model2 = torch.load(current_dir + "/" + decoder_save_path)
evaluateRandomly(
    model1.to(torch.device("cpu")), model2.to(torch.device("cpu")))
