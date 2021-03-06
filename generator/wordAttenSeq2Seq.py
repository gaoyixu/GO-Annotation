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
from encoder import AttenCombineEncoderRNN
from encoder import EncoderRNN
import datetime
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)
MAX_LENGTH = 100
word_max_length = 30
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.1
skiprate = 0.1


def train(input_tensors,
          target_tensor,
          encoder,
          decoder,
          CombineEncoder,
          encoder_optimizer,
          decoder_optimizer,
          CombineEncoder_optimizer,
          criterion,
          step,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    Combine_hidden = CombineEncoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    CombineEncoder_optimizer.zero_grad()
    input_size = len(input_tensors)
    target_length = target_tensor.size(0)
    loss = 0
    count = 0
    Combine_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    encoder_outputs = [torch.zeros(word_max_length, encoder.hidden_size, device=device) for i in range(max_length)]
    # encoder_outputs = torch.zeros(
    #     max_length, encoder.hidden_size, device=torch.device("cpu"))
    count = 0
    for i in range(min(input_size, MAX_LENGTH)):
        input_length = len(input_tensors[i])
        for ei in range(min(input_length,word_max_length)):
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

    decoder_input = torch.tensor([[SOS_token]]).type(torch.cuda.LongTensor)

    decoder_hidden = meanhiddens

    use_teaching_forcing = True if random.random(
    ) < teacher_forcing_ratio else False


    if use_teaching_forcing:
        for di in range(target_length):
            skip = True if random.random(
            ) < skiprate else False
            if skip:
                if di != target_length-1:
                    di += 1
                decoder_input = target_tensor[di]
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                     decoder_hidden,Combine_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            skip = True if random.random(
            ) < skiprate else False
            if skip:
                if di != target_length-1:
                    di += 1
                decoder_input = target_tensor[di]

            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                     decoder_hidden,Combine_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    if step % 200 == 0:
        print(step)
    if step % 5000 == 0:
        loss.backward()

    encoder_optimizer.step()
    CombineEncoder_optimizer.step()
    decoder_optimizer.step()


    return loss.item() / (target_length*(1-skiprate))


def trainIters(encoder,
               decoder,
               CombineEncoder,
               pairs,
               n_epoch,
               print_every=5000,
               plot_every=100,
               learning_rate=0.005):
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

    encoder_optimizer2 = torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.8, weight_decay=1e-5)
    decoder_optimizer2 = torch.optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.8, weight_decay=1e-5)
    CombineEncoder_optimizer2 = torch.optim.SGD(CombineEncoder.parameters(), lr=learning_rate, momentum=0.8,
                                                weight_decay=1e-5)
    encoder_optimizer2 = torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.8, weight_decay=1e-7)
    decoder_optimizer2 = torch.optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.8, weight_decay=1e-7)
    CombineEncoder_optimizer2 = torch.optim.SGD(CombineEncoder.parameters(), lr=learning_rate, momentum=0.8, weight_decay=1e-7)
    # randomPairs = [random.choice(pairs) for i in range(n_iter)]
    training_pairs = [
        tensorsFromPair(voc, pairs[i][2], pairs[i][0], "context")
        for i in range(len(pairs))
    ]
    criterion = nn.NLLLoss()

    pairLen = len(pairs)

    for i in range(n_epoch):
        random.shuffle(training_pairs)
        for iter in range(1, pairLen + 1):
            training_pair = training_pairs[iter - 1]
            input_tensors = training_pair[0]
            target_tensor = training_pair[1].type(torch.cuda.LongTensor)
            # if i > n_epoch*0.5:
            #     loss = train(input_tensors, target_tensor, encoder, decoder,CombineEncoder,
            #              encoder_optimizer1, decoder_optimizer1,CombineEncoder_optimizer2, criterion)
            # else:
            loss = train(input_tensors, target_tensor, encoder, decoder, CombineEncoder,
                            encoder_optimizer2, decoder_optimizer2, CombineEncoder_optimizer2, criterion,iter+i*pairLen)
            print_loss_total += loss
            plot_loss_total += loss

            if (iter+i*pairLen) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (iter+i*pairLen)/ (n_epoch*pairLen )), (iter+i*pairLen),
                                             (iter + i * pairLen)/ (n_epoch*pairLen) * 100, print_loss_avg))
                if print_loss_avg < 1.0:
                    break
            # if (iter+i*pairLen) % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0
            if (iter+i*pairLen) %(print_every*1) == 0:
                evaluateRandomly(encoder,decoder,CombineEncoder,pairs)
    showPlot(plot_losses)


def evaluate(encoder, decoder,CombineEncoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensors = [
            tensorFromSentence(voc, sentence[i]) for i in range(len(sentence))
        ]
        input_size = len(input_tensors)
        encoder_hidden = encoder.initHidden(device=torch.device("cuda"))
        Combine_hidden = CombineEncoder.initHidden(device=torch.device("cuda"))
        Combine_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_outputs = [torch.zeros(word_max_length, encoder.hidden_size, device=device) for i in range(max_length)]
        # encoder_outputs = torch.zeros(
        #     max_length, encoder.hidden_size, device=torch.device("cpu"))
        count = 0
        for i in range(min(input_size,MAX_LENGTH)):
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
                                                       decoder_hidden,Combine_outputs)
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


def evaluateRandomly(encoder, decoder,CombineEncoder,pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(pair)
        print('>', pair[2])
        print('=', pair[0])
        output_words,_ = evaluate(encoder, decoder,CombineEncoder, pair[2])
        score = bleu(pair[0].split(" "), output_words)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print("bleu", score)

def AttencombineEvaluate(encoder, decoder,CombineEncoder,voc, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensors = [
            tensorFromSentence(voc, sentence[i]) for i in range(len(sentence))
        ]
        input_size = len(input_tensors)
        encoder_hidden = encoder.initHidden(device=torch.device("cuda"))
        Combine_hidden = CombineEncoder.initHidden(device=torch.device("cuda"))
        Combine_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_outputs = [torch.zeros(word_max_length, encoder.hidden_size, device=device) for i in range(max_length)]

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

def combineEvaluateTotally(encoder, decoder,CombineEncoder,voc,pairs,length,model = "normal"):
    scoresSum = 0
    for i in range(length):
        pair = pairs[i]
        if model == "attention":
            output_words = AttencombineEvaluate(encoder, decoder,CombineEncoder,voc, pair[2])
        elif model == "wordattention":
            output_words = wordAttencombineEvaluate(encoder, decoder, CombineEncoder, voc, pair[2])
        else:
            output_words,_ = combineEvaluate(encoder, decoder, CombineEncoder, voc, pair[2])
        print('>', pair[2])
        print("=",pair[0])
        output_sentence = ' '.join(output_words)
        print("<",output_sentence)
        score = bleu(pair[0].split(" "), output_words[:-1])
        print(score)
        scoresSum += score
        # if i % 20 == 0 and i != 0:
        #     print("current round", i,"/",length,"bleu:","%.4f" % (scoresSum / i))
    print("total relu %.4f" % (scoresSum / length))
    return



nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H')
voc_path = abs_file_path + "/data/data_clean_lower.txt"
voc = Voc("total")
voc.initVoc(voc_path)
trainpairs, testpairs = prepareData(abs_file_path,"context")
# trainpairs = trainpairs[:100]
# testpairs = testpairs[:100]
descibe = [pair[2] for pair in trainpairs]
print("ontology name：",trainpairs[0][0],"\nontology token：",trainpairs[0][1],"\ngene desciption：",descibe[0],"\n",)
encodernum = max(map(lambda x:len(x),descibe))
print(encodernum)

hidden_size = 256
model = False
if model == True:
    day = 20
    hour = "08"
    nowTime = '2018-12-'+str(day)+'-'+str(hour)
    encoder_save_path = "model/combineEncoder+" +nowTime+ "+.pth"
    decoder_save_path = "model/combineDecoder+" +nowTime+ "+.pth"
    combiner_save_path = "model/combineCombiner+" +nowTime+ "+.pth"
    encoder1 = torch.load(current_dir + "/" + encoder_save_path)
    attn_decoder1 = torch.load(current_dir + "/" + decoder_save_path)
    CombineEncoder = torch.load(current_dir + "/" + combiner_save_path)
else:
    encoder1 = EncoderRNN(voc.num_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, voc.num_words).to(device)
    CombineEncoder = AttenCombineEncoderRNN(hidden_size, hidden_size).to(device)
trainIters(encoder1, attn_decoder1, CombineEncoder,trainpairs, 100)

encoder_save_path = "model/wordcombineEncoder+" +nowTime+ "hidden"+str(hidden_size)+".pth"
decoder_save_path = "model/wordcombineDecoder+" +nowTime+ "hidden"+str(hidden_size)+ "+.pth"
combiner_save_path = "model/wordcombineCombiner+" +nowTime+ "hidden"+str(hidden_size)+ "+.pth"
torch.save(encoder1, current_dir + '/' + encoder_save_path)
torch.save(attn_decoder1, current_dir + "/" + decoder_save_path)

torch.save(CombineEncoder, current_dir + "/" + combiner_save_path)
model1 = torch.load(current_dir + "/" + encoder_save_path)
model2 = torch.load(current_dir + "/" + decoder_save_path)
model3 = torch.load(current_dir + "/" + combiner_save_path)
# evaluateRandomly(
#     model1.to(device), model2.to(device),model3.to(device),testpairs)
combineEvaluateTotally(model1.to(device), model2.to(device),model3.to(device),voc,testpairs,len(testpairs),"wordattention")