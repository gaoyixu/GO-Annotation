from __future__ import print_function
import torch.nn as nn
import random
import os
from Voc import Voc
from util.datautil import *
from util.timeutil import *
from util.plotutil import *
from util.evalutil import *
from gcnfile.gcnutil import *
from decoder import AttnDecoderRNN
from encoder import doubleCombineEncoderRNN
from encoder import EncoderRNN
import datetime
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
abs_file_path = os.path.dirname(current_dir)
MAX_LENGTH = 397
SOS_token = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.2
lossRate = defaultdict(float)
step = 0




def trainIters(encoder,
               decoder,
               CombineEncoder,
               GCN,
               GcnEncoder,
               pairs,
               n_epoch,
               print_every=100,
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
    global teacher_forcing_ratio

    encoder_optimizer1 = torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.2, weight_decay=1e-8)
    decoder_optimizer1 = torch.optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.2, weight_decay=1e-8)
    CombineEncoder_optimizer1 = torch.optim.SGD(CombineEncoder.parameters(), lr=learning_rate, momentum=0.2,
                                                weight_decay=1e-8)
    GCN_optimizer1 = torch.optim.Adam(GCN.parameters(), lr= learning_rate*0.1, weight_decay=1e-8)
    GCNencoder_optimizer1 = torch.optim.SGD(GcnEncoder.parameters(), lr=learning_rate, momentum=0.2, weight_decay=1e-8)
    encoder_optimizer2 = torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
    decoder_optimizer2 = torch.optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
    CombineEncoder_optimizer2 = torch.optim.SGD(CombineEncoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
    GCN_optimizer2 = torch.optim.Adam(GCN.parameters(), lr=learning_rate*0.1, weight_decay=1e-5)
    GCNencoder_optimizer2 = torch.optim.SGD(GcnEncoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    #
    # encoder_optimizer3 = torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.85, weight_decay=1e-6)
    # decoder_optimizer3 = torch.optim.SGD(decoder.parameters(), lr=learning_rate, momentum=0.85, weight_decay=1e-6)
    # CombineEncoder_optimizer3 = torch.optim.SGD(CombineEncoder.parameters(), lr=learning_rate, momentum=0.85,
    #                                             weight_decay=1e-6)
    # randomPairs = [random.choice(pairs) for i in range(n_iter)]
    training_pairs = [
        tensorsFromPair(voc, pairs[i][2], pairs[i][0], "context")
        for i in range(len(pairs))
    ]

    # origin from the process of tensorFromSentence in datautil
    global tensor_word_frequency
    for key, value in tensor_word_frequency.items():
        lossRate[key] = min(maxF / value,10)/10
    print("the max rate",lossRate[max(lossRate, key=lossRate.get)])
    print("the min rate",lossRate[min(lossRate, key=lossRate.get)])
    print("of rate",lossRate[voc.word2index["of"]])
    maxRate = lossRate[max(lossRate, key=lossRate.get)]
    minRate = lossRate[min(lossRate, key=lossRate.get)]
    lossRate[EOS_token] = minRate
    # print(lossRate)


    criterion = nn.NLLLoss()

    pairLen = len(pairs)
    step = 0
    while(step<n_epoch) :
        random.shuffle(training_pairs)

        for iter in range(1, pairLen + 1):
            training_pair = training_pairs[iter - 1]
            input_tensors = training_pair[0]
            target_tensor = training_pair[1].type(torch.cuda.LongTensor)

            if step < 3:

                loss = train(input_tensors, target_tensor, encoder, decoder,CombineEncoder,GCN,GcnEncoder,
                 encoder_optimizer1, decoder_optimizer1,CombineEncoder_optimizer1,GCN_optimizer1,GCNencoder_optimizer1, criterion, iter)
            # elif i < n_epoch*0.6:
                # loss = train(input_tensors, target_tensor, encoder, decoder, CombineEncoder,
            #     #              encoder_optimizer2, decoder_optimizer2, CombineEncoder_optimizer2, criterion)
            else:
                # teacher_forcing_ratio = 0.1
                loss = train(input_tensors, target_tensor, encoder, decoder, CombineEncoder,GCN,GcnEncoder,
                            encoder_optimizer2, decoder_optimizer2, CombineEncoder_optimizer2,GCN_optimizer2,GCNencoder_optimizer2, criterion, iter)
            print_loss_total += loss
            plot_loss_total += loss

            if (iter+step*pairLen) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (iter+step*pairLen)/ (n_epoch*pairLen )), (iter+step*pairLen),
                                             (iter + step * pairLen)/ (n_epoch*pairLen) * 100, print_loss_avg))
                if print_loss_avg < 1.0:
                    break
            # if (iter+i*pairLen) % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0
            if (iter+step*pairLen) %(print_every*20) == 0:
                evaluateRandomly(encoder,decoder,CombineEncoder,GCN,GcnEncoder,pairs)
        step +=1
    showPlot(plot_losses)

def train(input_tensors,
          target_tensor,
          encoder,
          decoder,
          CombineEncoder,
          GCN,
          GcnEncoder,
          encoder_optimizer,
          decoder_optimizer,
          CombineEncoder_optimizer,
          GCN_optimizer,
          GCNencoder_optimizer,
          criterion,
          step,
          max_length=MAX_LENGTH
          ):
    encoder_hidden = encoder.initHidden()
    Combine_hidden = CombineEncoder.initHidden()
    gcn_encoder_hidden = GcnEncoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    CombineEncoder_optimizer.zero_grad()
    GCN_optimizer.zero_grad()
    GCNencoder_optimizer.zero_grad()
    GcnEncoder.train()
    Combine_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    input_size = len(input_tensors)
    target_length = target_tensor.size(0)
    loss = 0
    for i in range(min(input_size,MAX_LENGTH)):
        # encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
        input_length = len(input_tensors[i])
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensors[i][ei].type(torch.cuda.LongTensor),
                encoder_hidden)
            onehot_input = torch.FloatTensor(1,voc.num_words).zero_().scatter_(
                dim=1,index=torch.LongTensor([[input_tensors[i][ei].item()]]),value=1)
            gcn_output = GCN(
                onehot_input.type(torch.cuda.LongTensor),
                Laplace

            )
            gcn_encoder_output, gcn_encoder_hidden = GcnEncoder(
                gcn_output, encoder_hidden
            )


            # encoder_outputs[ei] = encoder_output[0, 0]
        # if i == 0:
        #     hiddens = encoder_hidden
        # else:
        #     hiddens = hiddens + encoder_hidden
        Combine_output, Combine_hidden = CombineEncoder(
            encoder_hidden.type(torch.cuda.FloatTensor),
            gcn_encoder_hidden.type(torch.cuda.FloatTensor),
            Combine_hidden
        )
        Combine_outputs[i] = Combine_output[0, 0]
    meanhiddens = Combine_hidden

    decoder_input = torch.tensor([[SOS_token]]).type(torch.cuda.LongTensor)

    decoder_hidden = meanhiddens

    use_teaching_forcing = True if random.random(
    ) < teacher_forcing_ratio else False

    if use_teaching_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                     decoder_hidden,Combine_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            # print(target_tensor[di].item())
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                     decoder_hidden,Combine_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                # while di < target_length:
                #     loss += criterion(decoder_output, target_tensor[di]) * lossRate[target_tensor[di].item()]
                #     di += 1
                break
    # print(step % 10 == 0)
    # if step % 10== 0:
    loss.backward()

    encoder_optimizer.step()
    CombineEncoder_optimizer.step()
    decoder_optimizer.step()
    GCN_optimizer.step()
    GCNencoder_optimizer.step()


    return loss.item() / target_length


def evaluate(encoder, decoder,CombineEncoder, GCN,GcnEncoder,sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensors = [
            tensorFromSentence(voc, sentence[i]) for i in range(len(sentence))
        ]
        input_size = len(input_tensors)
        encoder_hidden = encoder.initHidden(device=torch.device("cuda"))
        Combine_hidden = CombineEncoder.initHidden(device=torch.device("cuda"))
        gcn_encoder_hidden = GcnEncoder.initHidden(device=torch.device("cuda"))
        Combine_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        # encoder_outputs = torch.zeros(
        #     max_length, encoder.hidden_size, device=torch.device("cpu"))

        for i in range(min(input_size,MAX_LENGTH)):
            input_length = len(input_tensors[i])
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    input_tensors[i][ei].type(torch.cuda.LongTensor), encoder_hidden)
                encoder_hidden = encoder_hidden
                onehot_input = torch.FloatTensor(1, voc.num_words).zero_().scatter_(
                    dim=1, index=torch.LongTensor([[input_tensors[i][ei].item()]]), value=1)
                gcn_output = GCN(
                    onehot_input.type(torch.cuda.LongTensor),
                    Laplace
                )
                gcn_encoder_output, gcn_encoder_hidden = GcnEncoder(
                    gcn_output, encoder_hidden
                )
                # encoder_outputs[ei] += encoder_output[0, 0]
            # if i == 0:
            #     hiddens = encoder_hidden
            # else:
            #     hiddens = hiddens + encoder_hidden
            Combine_output, Combine_hidden = CombineEncoder(
                encoder_hidden.type(torch.cuda.FloatTensor),
                gcn_encoder_hidden.type(torch.cuda.FloatTensor),
                Combine_hidden
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


def evaluateRandomly(encoder, decoder,CombineEncoder,GCN,GcnEncoder,pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print(pair)
        print('>', pair[2])
        print('=', pair[0])
        output_words,_ = evaluate(encoder, decoder,CombineEncoder,GCN,GcnEncoder, pair[2])
        score = bleu(pair[0].split(" "), output_words)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print("bleu", score)


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
global tensor_word_frequency
for pair in trainpairs:
    indexes = [voc.word2index[word] for word in pair[0].split(" ")]

    for i in range(len(indexes)):
        # print(result[i])
        tensor_word_frequency[indexes[i]] += 1
maxF = tensor_word_frequency[max(tensor_word_frequency, key=tensor_word_frequency.get)]
tensor_word_frequency[EOS_token] = 1
print("maxF word",voc.index2word[max(tensor_word_frequency, key=tensor_word_frequency.get)])
print("the max frequency", maxF)

Laplace = initadj(voc,trainpairs,current_dir)
# Lambda , L  = preprocess(Laplace)
# print("A",A,"D",D)
Laplace = torch.tensor(Laplace).to(device)

# Lambda = torch.from_numpy(Lambda)
# Lambda = Lambda.type(torch.cuda.FloatTensor)
# LambdaL = torch.tensor(Lambda)
#
# L = torch.from_numpy(L)
# L = L.type(torch.cuda.FloatTensor)



model = False
hidden_size = 256
embedding_layer = nn.Embedding(voc.num_words, hidden_size)

if model == True:
    day = 19
    hour = "01"
    nowTime = '2018-12-'+str(day)+'-'+str(hour)
    encoder_save_path = "model/combineEncoder+" +nowTime+ "+.pth"
    decoder_save_path = "model/combineDecoder+" +nowTime+  "+.pth"
    combiner_save_path = "model/combineCombiner+" +nowTime+ "+.pth"
    gcn_save_path= "model/combinegcn+" +nowTime+ "+.pth"
    encoder1 = torch.load(current_dir + "/" + encoder_save_path)
    attn_decoder1 = torch.load(current_dir + "/" + decoder_save_path)
    CombineEncoder = torch.load(current_dir + "/" + combiner_save_path)
else:



    encoder1 = EncoderRNN(voc.num_words, hidden_size,embedding=embedding_layer, embedded=True).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, voc.num_words).to(device)
    CombineEncoder = doubleCombineEncoderRNN(hidden_size, hidden_size).to(device)
    GCN = GCN(voc.num_words, hidden_size, hidden=[12897], dropouts=[0.5]).to(device)
    GcnEncoder = EncoderRNN(hidden_size, hidden_size,embedding=embedding_layer, embedded=True).to(device)
trainIters(encoder1, attn_decoder1, CombineEncoder, GCN, GcnEncoder ,trainpairs,10)



encoder_save_path = "model/GCNcombineEncoder+" +nowTime+ "hidden"+str(hidden_size)+ "+.pth"
decoder_save_path = "model/GCNcombineDecoder+" +nowTime+ "hidden"+str(hidden_size)+ "+.pth"
combiner_save_path = "model/GCNcombineCombiner+" +nowTime+ "hidden"+str(hidden_size)+ "+.pth"
torch.save(encoder1, current_dir + '/' + encoder_save_path)
torch.save(attn_decoder1, current_dir + "/" + decoder_save_path)

torch.save(CombineEncoder, current_dir + "/" + combiner_save_path)
model1 = torch.load(current_dir + "/" + encoder_save_path)
model2 = torch.load(current_dir + "/" + decoder_save_path)
model3 = torch.load(current_dir + "/" + combiner_save_path)
# evaluateRandomly(
#     model1.to(device), model2.to(device),model3.to(device),testpairs)
combineEvaluateTotally(model1.to(device), model2.to(device),model3.to(device),voc,testpairs,len(testpairs),"attention")
print("this model is "+"model/AttencombineEncoder+" +nowTime+ "+.pth")