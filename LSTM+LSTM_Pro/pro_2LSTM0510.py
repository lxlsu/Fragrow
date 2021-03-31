import unicodedata
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import math
import random
import torch.nn as nn
from torch import optim
import time
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




teacher_forcing_ratio = 0.5
embedding_size = 3
embedding_pro_size = 3
encoder_hidden_size = 4


learning_rate = 0.005
print_every = 5000
plot_every = 500
n_epoches = 3
pro_hidden_size = 2
n_layers  = 3
dropout = 0.1
decoder_hidden_size = encoder_hidden_size

# n_iters = 10000

SOS_token, EOS_token = 0, 1
MAX_LENGTH = 27
MIN_LENGTH = 25
MAX_PROTEIN_LENGTH = 800

print('==========parameters===========')
print('learning rate : %s'% (learning_rate))
print('epoches : %s'% (n_epoches))
print('protein embedding size : %s'% (embedding_pro_size))
print('molecule embedding size : %s'% (embedding_size))
print('protein hidden size : %s'% (pro_hidden_size))
print('molecule hidden size : %s'% (encoder_hidden_size))
print('decoder hiddden size : %s'% (decoder_hidden_size))
print('dropout : %s'%(dropout))
print('layers : %s'%(n_layers))
print('plot_every : %s'% (plot_every))
print('max length : %s'%(MAX_LENGTH))
print('min length : %s'%(MIN_LENGTH))
print('说明')
print('===============================')

class Seq:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1}  # vocabulary
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # SOS, EOS

    def addSentence(self, sentence, mode):
        for word in sentence:
            self.addWord(word)
            if mode=='protein_lang' and  word =='2':
                print(sentence)
                #print(i)




    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    s = unicodeToAscii(s.strip())
    s = re.sub(r'\\', '_', s)  # 把序列里的'\'转义字符变成'_'

    return s


def initializeLangs(lang1, lang2, protein):
    print('Reading lines..')
    lines = open('../0727ligand_pro_15cutways.txt', encoding='utf-8').read().strip().split('\n')


    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    for p in pairs:
        comm = p[0]
        incomm = p[1]
        p[0] = incomm
        p[1] = comm
    input_lang = Seq(lang2)
    output_lang = Seq(lang1)
    protein_lang = Seq(protein)
    return input_lang, output_lang, protein_lang, pairs


def filterPair(p):
    return (len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH) and (len(p[0]) > MIN_LENGTH and len(p[1]) > MIN_LENGTH) and (len(p[2])<MAX_PROTEIN_LENGTH)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, protein):
    input_lang, output_lang, protein_lang, pairs = initializeLangs(lang1, lang2, protein)
    print('Read {} pairs'.format(len(pairs)))

    pairs = filterPairs(pairs)
    print('Trimmed to {} pairs'.format(len(pairs)))

    for pair in pairs:
        input_lang.addSentence(pair[0],'input_lang')
        output_lang.addSentence(pair[1],'output_lang')
        protein_lang.addSentence(pair[2],'protein_lang')
        #print(pair[2])
    '''
    for i in range(len(pairs)):
        input_lang.addSentence(pairs[i][0], 'input_lang',i)
        output_lang.addSentence(pairs[i][1], 'output_lang',i)
        protein_lang.addSentence(pairs[i][2], 'protein_lang',i)
    '''

    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(protein_lang.name, protein_lang.n_words)
    return input_lang, output_lang, protein_lang, pairs


input_lang, output_lang, protein_lang, pairs = prepareData('com', 'incom', 'protein')
print(random.choice(pairs))

print(input_lang.index2word)
print(output_lang.index2word)
print(protein_lang.index2word)
print("Cong! We have processed the data successfully!")
####################################################
#================seq2seq model======================

# LSTM Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,pro_hidden_size, n_layers, dropout):
        super(LSTMEncoder, self).__init__()
        # Parameters
        self.input_size = input_size  # 不完整分子序列字典词汇量
        self.embedding_size = embedding_size  # 嵌入层维数
        self.hidden_size = hidden_size  # LSTM单元的隐状态维数
        self.n_layers = n_layers  # LSTM单元层数
        self.dropout = dropout  # LSTM单元的dropout值

        # Layers Definition
        self.embedding = nn.Embedding(input_size, embedding_size)
        # 定义嵌入层
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=False,
                            batch_first=True)
        self.out = nn.Linear(pro_hidden_size, embedding_size)
        # 定义n_layers层的LSTM单元，bidirectional=False表示该单元为单向LSTM单元

    # 定义前向传播函数forward
    def forward(self, input, hidden, pro_output, i):
        # input为当前时间步输入的字符编码，为一维tensor
        # hidden为当前时间步LSTM单元的隐状态
        # 对于LSTM单元来说，由（hn,cn）构成，其维度均为(batch, num_layers*num_directions, hidden_size)
        # 由于设置batch_first=1，因此batch为输入输出的第一维
        # 在本模型中，每次喂Encoder和Decoder一个字符，batch = 1
        # 在本模型中，LSTM单元为单向，num_directions = 1
        if i==0:
            embedded = self.embedding(input).view(1, 1, -1)
            # 通过嵌入层将输入input嵌入为embedding_size维的tensor
            # 再通过view将其转换为三维tensor（1，1，embedding_size），以作为LSTM的输入
            output = embedded
            #print('==========#############')
            #print(output)
            output, hidden = self.lstm(output, hidden)
            # output = （batch = 1, seq_length = 1, num_directions*hidden_size = hidden_size)
            # hidden: hn,cn = (batch = 1, num_layers*num_directions, hidden_size)
        else:
            p_output = self.out(pro_output)
            #print(p_output)
            output, hidden = self.lstm(p_output, hidden)

        return output, hidden

    # initHidden函数用于初始化LSTM单元的hidden state
    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)



class LSTMProEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers, dropout):
        super(LSTMProEncoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers  # LSTM单元层数
        self.dropout = dropout  # LSTM单元的dropout值

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=False,
                            batch_first=True)


    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)



# LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, n_layers, dropout):
        super(LSTMDecoder, self).__init__()
        # Parameters
        self.output_size = output_size  # 完整分子序列字典词汇量
        self.embedding_size = embedding_size  # 嵌入层维数
        self.hidden_size = hidden_size  # LSTM单元的隐状态维数
        self.n_layers = n_layers  # LSTM单元层数
        self.dropout = dropout  # LSTM单元的dropout值


        # Layers Definition
        self.embedding = nn.Embedding(output_size, embedding_size)
        # 定义嵌入层
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=False,
                            batch_first=True)
        # 定义n_layers层的LSTM单元，bidirectional=False表示该单元为单向LSTM单元
        self.out = nn.Linear(hidden_size, output_size)
        # 定义全连接层out
        self.softmax = nn.LogSoftmax(dim=1)
        # 定义softmax层，选择LogSoftmax函数和后续train函数中计算loss的NLLLoss函数相组合

    def forward(self, input, hidden):
        # input为当前时间步输入的字符编码，为一维tensor
        # hidden为当前时间步LSTM单元的隐状态
        # 对于LSTM单元来说，由（hn,cn）构成，其维度均为(batch, num_layers*num_directions, hidden_size)
        # 在本模型中，每次喂Encoder和Decoder一个字符，batch = 1
        # 在本模型中，LSTM单元为单向，num_directions = 1
        output = self.embedding(input).view(1, 1, -1)
        # 通过嵌入层将输入input嵌入为embedding_size维的tensor
        # 再通过view将其转换为三维tensor（1，1，embedding_size），以作为LSTM的输入
        output = F.relu(output)
        # 选用ReLU激活函数
        output, hidden = self.lstm(output, hidden)
        # output = （batch = 1, seq_length = 1, num_directions*hidden_size = hidden_size)
        # hidden: hn,cn = (batch = 1, num_layers*num_directions, hidden_size）
        output = self.softmax(self.out(output[0]))
        # out层将output转换为tensor（1，output_size）
        # 其中，output_size为输出序列字典字符数
        return output, hidden

###############################################################
#========================train=================================


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '[Running time : %s | Leaving Time : %s ]' % (asMinutes(s), asMinutes(rs))


def showPlot(loss, axix):

    plt.title('Loss')
    plt.subplots_adjust(right=0.8)
    plt.plot(axix, loss)


    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    #plt.close()
    plt.savefig('results/LSTM-loss')
    plt.close()



def showPlot_val(loss, val_loss, axix):
    plt.title('Train Loss & Validate Loss')

    plt.subplots_adjust(right=0.8)

    plt.plot(axix, loss, label='Train Loss')

    plt.plot(axix, val_loss, label='Validate Loss')

    plt.legend()  # 显示图例

    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('results/LSTMloss-val_loss')
    plt.close()



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def Get_tensors(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    protein_tensor = tensorFromSentence(protein_lang,pair[2])
    return (input_tensor, target_tensor, protein_tensor)

# 将EncoderPro和EncoderMol输出的中间向量拼接得到Decoder的初始隐状态
def merge_hiddens(encoder_mol_hidden, encoder_pro_hidden):
    # encoder_mol_hidden为编码不完整分子序列的encoder最后一个时间步输出的隐状态
    # encoder_pro_hidden为编码蛋白质序列的encoder最后一个时间步输出的隐状态
    hiddens_n, cells_n = encoder_mol_hidden
    hiddens_n_pro, cells_n_pro = encoder_pro_hidden
    new_hiddens = torch.cat((hiddens_n_pro,hiddens_n),dim=-1)
    new_cells = torch.cat((cells_n_pro,cells_n),dim=-1)
    # 将隐状态中的hn和cn分别拼接起来，得到拼接后的（new_hiddens, new_cells），
    # 将其作为Decoder的初始隐状态
    return (new_hiddens, new_cells)


# 每run一次train函数，训练一条（不完整分子序列，完整分子序列，蛋白质序列）数据
def train(mol_tensor, protein_tensor, target_tensor,
          encoder_mol, encoder_pro, decoder,
          en_mol_optimizer, en_pro_optimizer, de_optimizer,
          criterion):
    # mol_tensor, protein_tensor, target_tensor分别为
    # 输入数据对中不完整分子序列，蛋白质序列，目标分子序列；
    # en_mol_optimizer, en_pro_optimizer, de_optimizer分别为优化器函数
    # criterion为选择的损失函数
    en_mol_optimizer.zero_grad()
    en_pro_optimizer.zero_grad()
    de_optimizer.zero_grad()
    # 将模型的参数梯度初始化为0

    loss = 0
    # 将loss初始化为0

    # ====================Encoder部分======================

    encoder_pro_hidden = (encoder_pro.initHidden().to(device), encoder_pro.initHidden().to(device))
    encoder_mol_hidden = (encoder_mol.initHidden().to(device), encoder_mol.initHidden().to(device))
    # 分别初始化两个encoder的hidden state

    mol_length = mol_tensor.size(0)
    pro_length = protein_tensor.size(0)
    print(pro_length)
    # mol_length, pro_length, target_length分别为输入数据对中不完整分子序列，蛋白质序列，目标分子序列长度


    # 逐字符输入蛋白质序列，通过EncoderPro将其编码为语义向量
    for pi in range(pro_length):
        encoder_pro_output, encoder_pro_hidden = encoder_pro(protein_tensor[pi], encoder_pro_hidden)

        # 逐字符输入不完整分子序列，通过EncoderMol将其编码为语义向量



    mol_output, encoder_mol_hidden = encoder_mol(encoder_pro_output, encoder_mol_hidden, encoder_pro_output, 1)
    for ei in range(mol_length):
        mol_output, encoder_mol_hidden = encoder_mol(mol_tensor[ei], encoder_mol_hidden, encoder_pro_output, 0)
        # mol_output = (1, 1, hidden_size))

        # encoder_mol_hidden = hn, cn = (1, num_layers_mol,  mol_hidden_size)

    #====================Decoder部分======================

    target_length = target_tensor.size(0)
    decoder_input = torch.tensor([[SOS_token]]).to(device)
    # 设置序列开始符SOS作为Decoder初始时间步输入
    decoder_hidden = encoder_mol_hidden
    # 将两个Encoder的中间向量concat起来，作为Decoder的初始隐状态
    # decoder_hidden = hn, cn = (1, num_layers,  mol_hidden_size+pro_hidden_size)

    # 每次输入字符时有0.5概率使用teacher forcing方法
    if_NTF = True if random.random() > teacher_forcing_ratio else False
    if if_NTF:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            # topv为输出的二维tensor中概率最大位的值，topi为索引index
            decoder_input = topi.squeeze().detach()
            # 不使用TF，用模型真实输出作为下一个时间步的输入

            loss += criterion(decoder_output, target_tensor[di])
            # 累加loss
            if decoder_input.item() == EOS_token:
                break
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # decoder_output = (1, 1, hidden_size))
            # decoder_hidden = hn, cn = (1, num_layers,  decoder_hidden_size)
            loss += criterion(decoder_output, target_tensor[di])
            # 累加 loss
            decoder_input = target_tensor[di]
            # teacher_forcing: 将数据集中的本时间步的目标输出target_tensor[di]作为下一时间步输入

    # 反向传播
    loss.backward()
    en_mol_optimizer.step()
    en_pro_optimizer.step()
    de_optimizer.step()
    return loss.item() / target_length #输出完整分子序列的平均loss值

def validate(input_tensor,protein_tensor, target_tensor, encoder,encoder_pro, decoder, criterion):

    target_length = target_tensor.size(0)
    input_length = input_tensor.size(0)
    pro_length = protein_tensor.size(0)

    loss = 0
    with torch.no_grad():

        encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
        # |encoder_hidden[0]|, |encoder_hidden[1]| = (num_layers*num_directions, batch_size, hidden_size/2)
        # encoder_outputs = torch.zeros(max_length + , encoder.hidden_size).to(device)
        # |encoder_outputs| = (max_length, hidden_size)
        encoder_pro_hidden = (encoder_pro.initHidden().to(device), encoder_pro.initHidden().to(device))



        ii = 0
        for pi in range(input_length, input_length + pro_length):
            encoder_pro_output, encoder_pro_hidden = encoder_pro(protein_tensor[ii], encoder_pro_hidden)
            ii = ii + 1
            #encoder_outputs[pi] = encoder_pro_output[0, 0]

        encoder_output, encoder_hidden = encoder(encoder_pro_output, encoder_hidden, encoder_pro_output, 1)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden,encoder_pro_output,0)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden



        #decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            #decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)
            loss += criterion(decoder_output, target_tensor[di])

            if topi.item() == EOS_token:
                break


            decoder_input = topi.squeeze().detach()

    return loss.item() / target_length


# 训练模型
def trainiters(pairs, encoder, encoder_pro, decoder, epoches, plot_loss=500, seed=1, learning_rate=0.005):
    #===============一些初始化================
    start = time.time()  # 训练开始时间
    plot_losses = []  # train-validate loss图中train loss
    plot_losses1 = []  # 用于画loss下降曲线
    plot_val_losses = []  # train-validate loss图中validate loss
    axix = []  # loss下降曲线横坐标
    bxix = []  # train validate loss下降曲线横坐标

    print_loss_total, plot_loss_total, plot_loss_total1 = 0, 0, 0
    print_loss_val_total, plot_loss_val_total = 0, 0

    pairs_tra_val, test_pairs = train_test_split(pairs, test_size=0.15, random_state= seed)
    train_pairs, val_pairs = train_test_split(pairs_tra_val, test_size=0.15, random_state= seed)
    # 分割训练集，验证集，测试集，为了比较不同模型，每次使用相同的seed，产生同样的训练集，验证集，测试集

    print('length of train pairs')
    print(len(train_pairs))
    print('length of test pairs')
    print(len(test_pairs))
    print('length of validate pairs')
    print(len(val_pairs))

    n_iters = epoches * len(train_pairs)  # 训练总次数
    iter_num = 0

    train_pairs = [Get_tensors(pair) for pair in train_pairs]
    val_pairs = [Get_tensors(pair) for pair in val_pairs]
    # 将训练集，验证集中的pairs向量化为（input_size,seq_length）向量，作为网络输入
    #print(val_pairs)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    encoder_pro_optimizer = optim.SGD(encoder_pro.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 多次实验尝试后，选择设定优化器使用SGD
    criterion = nn.NLLLoss()
    # loss函数选择负对数似然损失函数NLLLoss

    # 训练epoches个epoch
    # 每个epoch中，首先训练训练集中的所有数据，然后在验证集中进行验证
    # 每训练一个epoch，输出该epoch的平均loss，已用时间,预估剩余时间，在验证集上的loss
    # 每训练一个epoch，将训练后模型在验证集上进行验证
    # 每训练一个epoch，将这个epoch的平均train loss，validate loss分别作为一个坐标画图
    # 每训练过plot_loss个iteration时，将这段时间的平均train loss画在图上
    for epoch in range(1, epoches + 1):
        #==================train=================
        axix.append(epoch)
        # 添加x轴坐标
        len_train_pairs = len(train_pairs)
        for i in range(len_train_pairs):
            pair = train_pairs[i - 1]
            input_tensor, target_tensor, protein_tensor = pair[0], pair[1], pair[2]
            loss = train(input_tensor, protein_tensor, target_tensor, encoder, encoder_pro, decoder,
                         encoder_optimizer, encoder_pro_optimizer, decoder_optimizer, criterion)
            # 训练模型，输入一个（不完整分子序列，完整分子序列，蛋白质序列）数据对
            print_loss_total += loss
            plot_loss_total += loss
            plot_loss_total1 += loss
            # 累积loss
            iter_num = iter_num + 1
            # 迭代训练的总次数

            # 每训练plot_every次时，打印平均loss在loss图上
            if iter_num % plot_loss == 0:
                plot_loss_avg1 = plot_loss_total1 / plot_loss
                plot_losses1.append(plot_loss_avg1)
                plot_loss_total1 = 0
                bxix.append(iter_num)
        # 每个epoch输出一次这个epoch的平均loss
        print_loss_avg = print_loss_total / len(train_pairs)
        print_loss_total = 0
        # 每个epoch时将该epoch的平均train loss画出在 train validate loss图上
        plot_loss_avg = plot_loss_total / len(train_pairs)
        plot_losses.append(plot_loss_avg)
        #=====================validate===============
        len_val_pairs=len(val_pairs)
        for j in range(len_val_pairs):
            pair_val = val_pairs[j - 1]
            input_tensor_val, target_tensor_val, protein_tensor_val= pair_val[0], pair_val[1], pair_val[2]
            loss_val = validate(input_tensor_val,protein_tensor_val, target_tensor_val, encoder,encoder_pro, decoder, criterion)
            print_loss_val_total += loss_val
            plot_loss_val_total += loss_val

        print_loss_val_avg = print_loss_val_total / len(val_pairs)
        print_loss_val_total = 0
        plot_loss_val_avg = plot_loss_val_total / len(val_pairs)
        plot_val_losses.append(plot_loss_val_avg)
        plot_loss_total = 0
        plot_loss_val_total = 0
        print('Epoch: %d : %s, Iterations : %d (%d%%) Train Loss: %.4f | Val Loss: %.4f' % (epoch, timeSince(start, iter_num / n_iters),
                                                       iter_num, iter_num / n_iters * 100, print_loss_avg, print_loss_val_avg))
        # 每个epoch进行一次输出
    showPlot_val(plot_losses, plot_val_losses, axix)
    showPlot(plot_losses1, bxix)
    # 分别画出两个loss曲线图
    torch.save(encoder.state_dict(), 'results/encoder.pth')
    torch.save(encoder_pro.state_dict(), 'results/encoder_pro.pth')
    torch.save(decoder.state_dict(), 'results/decoder.pth')
    # 保存训练后模型


######################training parameters##########################################




#input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

encoder = LSTMEncoder(input_size=input_lang.n_words,
                              embedding_size=embedding_size,pro_hidden_size=pro_hidden_size,
                              hidden_size=encoder_hidden_size, n_layers=n_layers,dropout=dropout
                              ).to(device)

encoder_pro = LSTMProEncoder(input_size=protein_lang.n_words,
                              embedding_size=embedding_pro_size,
                              hidden_size=pro_hidden_size, n_layers=n_layers,dropout=dropout
                              ).to(device)

decoder = LSTMDecoder(output_size=output_lang.n_words,
                              embedding_size=embedding_size,
                              hidden_size=decoder_hidden_size, n_layers=n_layers,dropout=dropout
                              ).to(device)

trainiters(pairs, encoder, encoder_pro, decoder, epoches=n_epoches,learning_rate=learning_rate, plot_loss=plot_every)


###########################evaluate############################

def FlitBack(s): #转换回最初的SMILES分子式，将'_'转换回'\'
    b = re.sub('_', r'\\', s)
    return b



def translate(pair, output):
    file_object = open('results/LSTMresults.txt', 'a', encoding='utf-8')
    file_object.writelines(FlitBack(pair[0]) + '\t')
    file_object.writelines(FlitBack(pair[1]) + '\t')
    file_object.writelines(FlitBack(output) + '\n')


def evaluate(sentence, protein, encoder, encoder_pro, decoder, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        # |input_tensor| = (sentence_length, 1)
        input_length = input_tensor.size(0)

        input_protein = tensorFromSentence(protein_lang, protein)
        pro_length = input_protein.size(0)

        encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
        encoder_pro_hidden = (encoder_pro.initHidden().to(device), encoder_pro.initHidden().to(device))

        ii = 0
        for pi in range(input_length, input_length + pro_length):
            encoder_pro_output, encoder_pro_hidden = encoder_pro(input_protein[ii], encoder_pro_hidden)
            ii = ii + 1

        encoder_output, encoder_hidden = encoder(encoder_pro_output, encoder_hidden, encoder_pro_output, 1)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden,encoder_pro_output,0)

        decoder_input = torch.tensor([[SOS_token]]).to(device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)  # top-1 value, index


            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

    return decoded_words


def evaluateiters(pairs, encoder, encoder_pro, decoder, train_pairs_seed=1):
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)

    for pi, pair in enumerate(test_pairs):
        output_words = evaluate(pair[0], pair[2], encoder, encoder_pro, decoder)
        output_sentence = ''.join(output_words)

        # for print
        translate(pair, output_sentence)

    print('evaluate successfully')


encoder = LSTMEncoder(input_size=input_lang.n_words,
                              embedding_size=embedding_size,pro_hidden_size=pro_hidden_size,
                              hidden_size=encoder_hidden_size,n_layers=n_layers,dropout=dropout
                              ).to(device)

encoder_pro = LSTMProEncoder(input_size=protein_lang.n_words,
                              embedding_size=embedding_pro_size,
                              hidden_size=pro_hidden_size,n_layers=n_layers,dropout=dropout
                              ).to(device)

decoder = LSTMDecoder(output_size=output_lang.n_words,
                              embedding_size=embedding_size,
                              hidden_size=decoder_hidden_size,n_layers=n_layers,dropout=dropout
                              ).to(device)


evaluateiters(pairs, encoder, encoder_pro, decoder)

