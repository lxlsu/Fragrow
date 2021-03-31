import unicodedata
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import math
import random
import torch
import torch.nn as nn
from torch import optim
import time
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from mpl_toolkits.axes_grid1 import host_subplot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5
embedding_mol_size = 3
embedding_pro_size = 3


dropout=0.1
n_layers= 3
learning_rate = 0.005
plot_every = 500
n_epoches = 3
pro_hidden_size = 2
encoder_hidden_size = 4
decoder_hidden_size = encoder_hidden_size + pro_hidden_size

# n_iters = 10000

SOS_token, EOS_token = 0, 1
MAX_LENGTH = 27
MIN_LENGTH = 25
MAX_PROTEIN_LENGTH = 800

print('==========parameters===========')
print('learning rate : %s'% (learning_rate))
print('epoches : %s'% (n_epoches))
print('protein embedding size : %s'% (embedding_pro_size))
print('molecule embedding size : %s'% (embedding_mol_size))
print('protein hidden size : %s'% (pro_hidden_size))
print('molecule hidden size : %s'% (encoder_hidden_size))
print('decoder hiddden size : %s'% (decoder_hidden_size))
print('dropout : %s'%(dropout))
print('layers : %s'%(n_layers))
print('plot_every : %s'% (plot_every))
print('max length : %s'%(MAX_LENGTH))
print('min length : %s'%(MIN_LENGTH))
print('说明 ')
print('===============================')



class Seq:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1}  # vocabulary
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # SOS, EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

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


def readLangs(lang1, lang2, protein):
    print('Reading lines..')
    # Read the file and split into lines
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
    return (len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH) and (len(p[0]) > MIN_LENGTH and len(p[1]) > MIN_LENGTH)and (len(p[2])<MAX_PROTEIN_LENGTH)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, protein):
    input_lang, output_lang, protein_lang, pairs = readLangs(lang1, lang2, protein)
    print('Read {} pairs'.format(len(pairs)))

    pairs = filterPairs(pairs)
    print('Trimmed to {} pairs'.format(len(pairs)))

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        protein_lang.addSentence(pair[2])

    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(protein_lang.name, protein_lang.n_words)
    return input_lang, output_lang, protein_lang, pairs


input_lang, output_lang, protein_lang, pairs = prepareData('com', 'incom', 'protein')
print(random.choice(pairs))

print(input_lang.index2word)
print(protein_lang.index2word)
print("Cong! We have processed the data successfully!")
####################################################
#================seq2seq model======================

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 n_layers=3, dropout_p=.1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, int(hidden_size / 2),
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers * 2, 1, int(self.hidden_size / 2))


class EncoderPro(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 n_layers=3, dropout_p=.1):
        super(EncoderPro, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, int(hidden_size / 2),
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers * 2, 1, int(self.hidden_size / 2))

# 添加attention机制的Decoder模型
class AttnDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, encoder_hidden_size, pro_hidden_size,
                  n_layers=n_layers, dropout_p=dropout):
        super(AttnDecoder, self).__init__()
        # Parameters
        self.output_size = output_size  # 完整分子序列字典词汇量
        self.embedding_size = embedding_size  # 嵌入层维数
        self.hidden_size = hidden_size  # LSTM单元的隐状态维数
        self.n_layers =  n_layers  # LSTM单元层数
        self.dropout = dropout_p  # LSTM单元的dropout值
        self.encoder_hidden_size = encoder_hidden_size
        # 编码不完整分子序列的encoder的hidden size
        self.pro_hidden_size = pro_hidden_size
        # 编码蛋白质序列的encoder的hidden size

        # Layers Definition
        self.embedding = nn.Embedding(output_size, embedding_size)
        # 定义嵌入层

        self.out_mol = nn.Linear(self.hidden_size, self.encoder_hidden_size)
        self.out_pro = nn.Linear(self.hidden_size, self.pro_hidden_size)
        # 定义两个全连接层，用于将tensor转换为encoder_hidden_size维和pro_hidden_size维，
        # 便于后续计算attention权重

        self.lstm = nn.LSTM(embedding_size, hidden_size,num_layers=n_layers,dropout=dropout_p,
                            bidirectional=False,batch_first=True)
        # Decoder的RNN单元选择多层单向LSTM
        self.out = nn.Linear(self.hidden_size + self.encoder_hidden_size + self.pro_hidden_size, self.output_size)
        # 定义一个全连接层，将输入的向量转换为output_size维向量，用以输出
        self.softmax = nn.LogSoftmax(dim=1)
        # 定义softmax层，选择LogSoftmax函数和后续train函数中计算loss的NLLLoss函数相组合

    # 定义前向传播函数forward
    def forward(self, input, hidden, encoder_outputs, pro_outputs):
        # input为当前时间步输入的字符编码，为一维tensor
        # hidden为当前时间步LSTM单元的隐状态
        # 对于LSTM单元来说，由（hn,cn）构成，其维度均为(batch, num_layers*num_directions, hidden_size)
        # 在本模型中，每次喂Encoder和Decoder一个字符，batch = 1
        # 在本模型中，Decoder的LSTM单元为单向，num_directions = 1
        # encoder_outputs是编码不完整分子序列的encoder的全部输出，大小为(max_mol_length, encoder_hidden_size)
        # pro_outputs是编码蛋白质序列的encoder的全部输出，大小为(max_pro_length, pro_hidden_size)
        output = self.embedding(input).view(1, 1, -1)
        # 通过嵌入层将输入input嵌入为embedding_size维的tensor
        # 再通过view将其转换为三维tensor（1，1，embedding_size），以作为LSTM的输入
        output = F.relu(output)
        # 选用ReLU激活函数

        output, hidden = self.lstm(output, hidden)
        # output = （batch = 1, seq_length = 1, num_directions*hidden_size = hidden_size)
        # hidden: hn,cn = (batch = 1, num_layers*num_directions, hidden_size）

        output_encoder = self.out_mol(output[0])
        #print(output_encoder)
        output_encoder = output_encoder.unsqueeze(0)
        # 通过全连接层将Decoder的output转换为（1,1,encoder_hidden_size），
        # 用于后续和encoder_outputs进行矩阵乘法，计算注意力权重

        output_pro = self.out_pro(output[0])
        output_pro = output_pro.unsqueeze(0)
        # 通过全连接层将Decoder的output转换为（1,1,pro_hidden_size），
        # 用于后续和pro_outputs进行矩阵乘法，计算注意力权重

        attn_weights_mol = torch.bmm(output_encoder, encoder_outputs.unsqueeze(0).permute(0, 2, 1)).squeeze(0)
        #print(attn_weights_mol)
        attn_weights_pro = torch.bmm(output_pro, pro_outputs.unsqueeze(0).permute(0, 2, 1)).squeeze(0)
        # 将转换维度后的Decoder output分别与encoder_outputs, pro_outputs矩阵相乘，分别得到注意力权重

        attn_weights_mol = F.softmax(attn_weights_mol, dim=1)
        attn_weights_pro = F.softmax(attn_weights_pro, dim=1)
        # 将得到的注意力权重通过softmax函数转换为概率分布
        # 分别得到当前时间步对不完整分子序列，蛋白质序列的注意力权重概率分布
        # 其size分别为 (1, max_mol_length)(1, max_pro_length)
        #print(attn_weights_mol)

        context_mol = torch.bmm(attn_weights_mol.unsqueeze(0), encoder_outputs.unsqueeze(0))
        context_pro = torch.bmm(attn_weights_pro.unsqueeze(0), pro_outputs.unsqueeze(0))
        # 将两个encoder的输出分别与注意力权重相乘，分别对mol和pro得到两个语义向量
        # contect_mol = (batch_size=1, 1, hidden_size)
        # contect_pro = (batch_size=1, 1, hidden_size)
        #print(context_mol)

        output = torch.cat((output, context_mol, context_pro), -1)
        # 将得到的两个语义向量和decoder的output拼接起来
        # output = （batch_size=1, 1, hidden_size + encoder_hidden_size + pro_hidden_size）

        output = self.softmax(self.out(output[0]))
        # 通过全连接层将上一部得到的向量转换为output_size维，
        # 然后通过logsoftmax函数将其转换为概率分布
        # output = (1, output_size)

        return output, hidden, attn_weights_mol, attn_weights_pro
        #  输出当前时间步输出，隐状态，对不完整分子序列和蛋白质序列的注意力权重


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(loss, axix):

    plt.title('Loss')
    #host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    # par1 = host.twinx()  # 共享x轴

    #plt.title('Loss')
    plt.plot(axix, loss)

    #plt.plot(axix, val_loss, label='Validate Loss')

    #plt.legend()  # 显示图例

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    #plt.close()
    plt.savefig('results/attnBiLSTM-loss')
    plt.close()



def showPlot_val(loss, val_loss, axix):
    plt.title('Train Loss & Validate Loss')
    #host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    #par1 = host.twinx()  # 共享x轴



    plt.plot(axix, loss, label='Train Loss')

    plt.plot(axix, val_loss, label='Validate Loss')

    plt.legend()  # 显示图例

    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('results/loss-val_loss')
    plt.close()


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    protein_tensor = tensorFromSentence(protein_lang, pair[2])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor, protein_tensor)


def merge_encoder_hiddens(encoder_hiddens):
    new_hiddens, new_cells = [], []
    hiddens, cells = encoder_hiddens
    for i in range(0, hiddens.size(0), 2):
        new_hiddens += [torch.cat([hiddens[i], hiddens[i + 1]], dim=-1)]
        new_cells += [torch.cat([cells[i], cells[i + 1]], dim=-1)]

    new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)
    return (new_hiddens, new_cells)

def merge_encoder_pro_hiddens(encoder_hidden, encoder_pro_hidden):
    hiddens_en, cells_en = encoder_hidden
    hiddens_pro, cells_pro = encoder_pro_hidden
    new_hiddens = torch.cat((hiddens_pro, hiddens_en), dim=-1)
    new_cells = torch.cat((cells_pro, cells_en), dim=-1)
    return (new_hiddens, new_cells)


def train(input_tensor, protein_tensor, target_tensor,
          encoder, encoder_pro, decoder,
          encoder_optimizer, encoder_pro_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH, max_pro_length=MAX_PROTEIN_LENGTH+1):
    encoder_optimizer.zero_grad()
    encoder_pro_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length, target_length = input_tensor.size(0), target_tensor.size(0)
    protein_length = protein_tensor.size(0)

    encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

    encoder_pro_hidden = (encoder_pro.initHidden().to(device), encoder_pro.initHidden().to(device))
    encoder_pro_outputs = torch.zeros(max_pro_length, encoder_pro.hidden_size).to(device)


    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0, 0]

    for pi in range(protein_length):
        encoder_pro_output, encoder_pro_hidden = encoder_pro(protein_tensor[pi], encoder_pro_hidden)

        encoder_pro_outputs[pi] = encoder_pro_output[0, 0]

    encoder_fin_hidden = merge_encoder_hiddens(encoder_hidden)
    encoder_pro_fin_hidden = merge_encoder_hiddens(encoder_pro_hidden)

    decoder_input = torch.tensor([[SOS_token]]).to(device)
    decoder_hidden = merge_encoder_pro_hiddens(encoder_fin_hidden, encoder_pro_fin_hidden)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention_mol, decoder_attention_pro = decoder(decoder_input,
                                                                        decoder_hidden, encoder_outputs, encoder_pro_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention_mol, decoder_attention_pro = decoder(decoder_input,
                                                                        decoder_hidden, encoder_outputs, encoder_pro_outputs)


            topv, topi = decoder_output.topk(1)

            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    encoder_pro_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def velidate(input_tensor, protein_tensor, target_tensor,
             encoder, encoder_pro, decoder, criterion, max_length=MAX_LENGTH, max_pro_length=MAX_PROTEIN_LENGTH+1):

    target_length = target_tensor.size(0)

    loss = 0
    with torch.no_grad():
        input_length = input_tensor.size(0)
        protein_length = protein_tensor.size(0)


        encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

        encoder_pro_hidden = (encoder_pro.initHidden().to(device), encoder_pro.initHidden().to(device))
        encoder_pro_outputs = torch.zeros(max_pro_length, encoder_pro.hidden_size).to(device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        for pi in range(protein_length):
            encoder_pro_output, encoder_pro_hidden = encoder_pro(protein_tensor[pi], encoder_pro_hidden)
            encoder_pro_outputs[pi] += encoder_pro_output[0, 0]

        encoder_fin_hidden = merge_encoder_hiddens(encoder_hidden)
        encoder_pro_fin_hidden = merge_encoder_hiddens(encoder_pro_hidden)


        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = merge_encoder_pro_hiddens(encoder_fin_hidden, encoder_pro_fin_hidden)

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention_mol,decoder_pro_attention = decoder(decoder_input,
                                                                        decoder_hidden, encoder_outputs, encoder_pro_outputs)


            topv, topi = decoder_output.data.topk(1)
            loss += criterion(decoder_output, target_tensor[di])

            if topi.item() == EOS_token:
                break

            decoder_input = topi.squeeze().detach()

    return loss.item() / target_length



def trainiters(pairs, encoder, encoder_pro, decoder,  epoches, plot_every=500,
               train_pairs_seed=1, learning_rate=.01):

    start = time.time()
    plot_losses = []
    plot_losses1 = []
    plot_val_losses = []
    #plot_losses1=
    axix = []
    bxix = []

    print_loss_total, plot_loss_total, plot_loss_total1 = 0, 0, 0
    print_loss_val_total, plot_loss_val_total = 0, 0
    #best_valid_loss = 1000000

    pairs_tra_val, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)
    train_pairs, val_pairs = train_test_split(pairs_tra_val, test_size=0.15, random_state=train_pairs_seed)

    n_iters = epoches * len(train_pairs)
    iter_num = 0

    train_pairs = [tensorsFromPair(pair) for pair in train_pairs]

    val_pairs = [tensorsFromPair(pair) for pair in val_pairs]

    print('length of train pairs')
    print(len(train_pairs))
    print('length of test pairs')
    print(len(test_pairs))
    print('length of validate pairs')
    print(len(val_pairs))


    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    encoder_pro_optimizer = optim.SGD(encoder_pro.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, epoches + 1):
        axix.append(epoch)
        i = len(train_pairs)
        for i in range(i):
            pair = train_pairs[i - 1]
            input_tensor, target_tensor, protein_tensor = pair[0], pair[1], pair[2]
            loss = train(input_tensor, protein_tensor, target_tensor, encoder, encoder_pro, decoder,
                         encoder_optimizer, encoder_pro_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            plot_loss_total1 += loss
            iter_num = iter_num + 1  # 迭代训练的总次数


            if iter_num % plot_every == 0:
                plot_loss_avg1 = plot_loss_total1 / plot_every
                plot_losses1.append(plot_loss_avg1)
                plot_loss_total1 = 0
                bxix.append(iter_num)



        print_loss_avg = print_loss_total / len(train_pairs)
        print_loss_total = 0

        plot_loss_avg = plot_loss_total / len(train_pairs)
        plot_losses.append(plot_loss_avg)

        j=len(val_pairs)
        for j in range(j):
            pair_val = val_pairs[j - 1]
            input_tensor_val, target_tensor_val, protein_tensor_val = pair_val[0], pair_val[1], pair_val[2]
            loss_val = velidate(input_tensor_val, protein_tensor_val, target_tensor_val, encoder, encoder_pro, decoder, criterion)
            print_loss_val_total += loss_val
            plot_loss_val_total += loss_val

        print_loss_val_avg = print_loss_val_total / len(val_pairs)
        print_loss_val_total = 0

        plot_loss_val_avg = plot_loss_val_total / len(val_pairs)
        plot_val_losses.append(plot_loss_val_avg)

        plot_loss_total = 0
        plot_loss_val_total = 0

        print('Epoch: %d : %s (%d %d%%) Loss: %.4f | Val: %.4f' % (epoch, timeSince(start, iter_num / n_iters),
                                                       iter_num, iter_num / n_iters * 100, print_loss_avg, print_loss_val_avg))



    showPlot_val(plot_losses, plot_val_losses, axix)
    showPlot(plot_losses1, bxix)



    torch.save(encoder.state_dict(), 'results/encoder.pth')
    torch.save(encoder_pro.state_dict(), 'results/encoder_pro.pth')
    torch.save(decoder.state_dict(), 'results/decoder.pth')




encoder = Encoder(input_size = input_lang.n_words,
                              embedding_size = embedding_mol_size,
                              hidden_size = encoder_hidden_size,
                              n_layers = n_layers,
                              dropout_p = dropout
                              ).to(device)

encoder_pro = EncoderPro(input_size = protein_lang.n_words,
                              embedding_size = embedding_pro_size,
                              hidden_size = pro_hidden_size,
                              n_layers = n_layers,
                              dropout_p = dropout
                              ).to(device)

decoder = AttnDecoder(output_size = output_lang.n_words,
                                  embedding_size = embedding_mol_size,
                                  hidden_size = decoder_hidden_size, encoder_hidden_size = encoder_hidden_size,
                                  n_layers = n_layers,
                                  dropout_p = dropout, pro_hidden_size = pro_hidden_size
                                  ).to(device)

trainiters(pairs, encoder, encoder_pro, decoder, n_epoches,plot_every=plot_every, learning_rate=learning_rate)

def FlitBack(s): #转换回最初的SMILES分子式，将'_'转换回'\'
    b = re.sub('_', r'\\', s)
    return b

def translate(pair, output):
    file_object = open('results/AttnProBiLSTMresults.txt', 'a', encoding='utf-8')

    file_object.writelines(FlitBack(pair[0]) + '\t')
    file_object.writelines(FlitBack(pair[1]) + '\t')
    file_object.writelines(FlitBack(output) + '\n')


def evaluate(sentence, protein, encoder, encoder_pro, decoder, max_length=MAX_LENGTH, max_pro_length=MAX_PROTEIN_LENGTH+1):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size(0)

        protein_tensor = tensorFromSentence(protein_lang, protein)
        protein_length = protein_tensor.size(0)

        encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

        encoder_pro_hidden = (encoder_pro.initHidden().to(device), encoder_pro.initHidden().to(device))
        encoder_pro_outputs = torch.zeros(max_pro_length, encoder_pro.hidden_size).to(device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        for pi in range(protein_length):
            encoder_pro_output, encoder_pro_hidden = encoder_pro(protein_tensor[pi], encoder_pro_hidden)
            encoder_pro_outputs[pi] += encoder_pro_output[0, 0]

        encoder_fin_hidden = merge_encoder_hiddens(encoder_hidden)
        encoder_pro_fin_hidden = merge_encoder_hiddens(encoder_pro_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = merge_encoder_pro_hiddens(encoder_fin_hidden, encoder_pro_fin_hidden)

        decoded_words = []
        decoder_attentions_mol = torch.zeros(max_length, input_length)
        decoder_attentions_pro = torch.zeros(max_length, protein_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention_mol, decoder_attention_pro = decoder(decoder_input,
                                                                        decoder_hidden, encoder_outputs, encoder_pro_outputs)

            decoder_attentions_mol[di] = decoder_attention_mol.data
            decoder_attentions_pro[di] = decoder_attention_pro.data

            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

    return decoded_words, decoder_attentions_mol[:di + 1], decoder_attentions_pro[:di + 1]

train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=1)

def evaluateiters(pairs, encoder, encoder_pro, decoder, train_pairs_seed=1):

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)

    for pi, pair in enumerate(test_pairs):
        output_words, decoder_attn, decoder_attn_pro = evaluate(pair[0], pair[2], encoder, encoder_pro, decoder)
        output_sentence = ''.join(output_words)

        # for print
        translate(pair, output_sentence)

evaluateiters(pairs, encoder, encoder_pro, decoder)


def visual_attention(input_sentence, output_words, attentions, i, mode):

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    im = ax.matshow(attentions.numpy(), cmap='bone')

    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    fig.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax = cax)
    #fig.colorbar(cax)

    sepp=[]
    seq_length = len(input_sentence)
    for seq in range(seq_length):
        sepp.append(input_sentence[seq])

    # Set up axes
    ax.set_xticklabels([''] + sepp + ['*'])
    ax.set_yticklabels([''] + output_words+ ['*'])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig('results/attention_%s_%s' % (mode, i))
    plt.close()

#随机在测试集取20个数据进行测试
def rondom_evaluate_and_show_attention():
    for i in range(20):
        pair = random.choice(test_pairs)
        output_words, decoder_attn_mol, decoder_attn_pro = evaluate(pair[0], pair[2], encoder, encoder_pro, decoder)
        print('input =', pair[0])
        print('answer=', pair[1])
        print('output =', ''.join(output_words))
        print('\n')
        visual_attention(pair[0], output_words, decoder_attn_mol, i, mode='mol')
        visual_attention(pair[2], output_words, decoder_attn_pro, i, mode='pro')
        #print(decoder_attn)
        #print(torch.sum(decoder_attn))


rondom_evaluate_and_show_attention()

