import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import time
import math
import random
import modbus_tk
import modbus_tk.modbus_tcp as modbus_tcp
import check_tcp as ch

LOGGER = modbus_tk.utils.create_logger("console")
Epoch=[]
Tran_loss=[]
Val_loss=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('./modbus_tcp.txt', 'r', encoding='utf-8') as f:
    data = f.read()
data = data.strip()
data = data.split('\n')

en_data = [line.split(' ')[0] for line in data]
ch_data = [line.split(' ')[1] for line in data]
# print("en_data",en_data)
# print("ch_data",ch_data)
en_token_list =[ ]
ch_token_list =[ ]
for str1 in en_data:
    data1 = []
    for i in range(0, len(str1), 2):
        data1.append(str1[i]+str1[i + 1])
    data1.append("<eos>")
    en_token_list.append(data1)
for str2 in ch_data:
    data2=[]
    for i in range(0, len(str2), 2):
        data2.append(str2[i] + str2[i + 1])
    data2.append("<eos>")
    ch_token_list.append(data2)
basic_dict = {'<bos>': 256, '<eos>': 257, '<pad>': 258,"<unk>":259}
def  build_key():
    key = {}
    str1 = ['a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    basic_dict = {'<bos>': 256, '<eos>': 257, '<pad>': 258,"<unk>":259}
    n = 0
    for x in str1:
        for y in str1:
           keys = x+y
           key[keys] = n
           n= n + 1
    key.update(basic_dict)
    return key
key = build_key()
key1 = build_key()

src_vocab_size = len(key)
idx2word = {i: w for i, w in enumerate(key)}

ch_num_data = [[key[ch] for ch in line] for line in ch_token_list]
en_num_data = [[key[en] for en in line ] for line in en_token_list]


class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data
        #
        # assert len(src_data) == len(trg_data), \
        #     "numbers of src_data  and trg_data must be equal!"

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_sample =self.src_data[idx]
        src_len = len(self.src_data[idx])
        trg_sample = self.trg_data[idx]
        trg_len = len(self.trg_data[idx])
        return {"src": src_sample, "src_len": src_len, "trg": trg_sample, "trg_len": trg_len}


def padding_batch(batch):
    src_lens = [d["src_len"] for d in batch]
    trg_lens = [d["trg_len"] for d in batch]

    src_max = max([d["src_len"] for d in batch])
    trg_max = max([d["trg_len"] for d in batch])
    for d in batch:
        d["src"].extend([key["<pad>"]] * (src_max - d["src_len"]))
        d["trg"].extend([key["<pad>"]] * (trg_max - d["trg_len"]))
    srcs = torch.tensor([pair["src"] for pair in batch], dtype=torch.long, device=device)
    trgs = torch.tensor([pair["trg"] for pair in batch], dtype=torch.long, device=device)

    batch = {"src": srcs.T, "src_len": src_lens, "trg": trgs.T, "trg_len": trg_lens}
    #print(batch)
    return batch


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True):
        super(Encoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_seqs, input_lengths, hidden):
        # input_seqs = [seq_len, batch]
        embedded = self.embedding(input_seqs)
        # embedded = [seq_len, batch, embed_dim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)

        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs = [seq_len, batch, hid_dim * n directions]
        # output_lengths = [batch]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

        if bidirectional:
            self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        else:
            self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, token_inputs, hidden):
        # token_inputs = [batch]
        batch_size = token_inputs.size(0)
        embedded = self.dropout(self.embedding(token_inputs).view(1, batch_size, -1))
        # embedded = [1, batch, emb_dim]

        output, hidden = self.gru(embedded, hidden)
        # output = [1, batch,  n_directions * hid_dim]
        # hidden = [n_layers * n_directions, batch, hid_dim]

        output = self.fc_out(output.squeeze(0))
        output = self.softmax(output)
        # output = [batch, output_dim]
        return output, hidden

def softmax1(data):
    data = list(data)
    denominator = 0
    j = 1
    l = len(data)
    for i in data:
        denominator = denominator + math.exp(i / j)
    for i in range(0, l):
        data[i] = math.exp(data[i] / j) / denominator
    return data

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device,predict=False,basic_dict=None,max_len=100):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder

        self.predict = predict  # 训练阶段还是预测阶段
        self.basic_dict = basic_dict  # decoder的字典，存放特殊token对应的id
        self.max_len = max_len  # 翻译时最大输出长度

        self.enc_n_layers = self.encoder.gru.num_layers
        self.enc_n_directions = 2 if self.encoder.gru.bidirectional else 1
        self.dec_n_directions = 2 if self.decoder.gru.bidirectional else 1

    def forward(self, input_batches, input_lengths, target_batches=None, target_lengths=None,
                teacher_forcing_ratio=0.5):
        # input_batches = target_batches = [seq_len, batch]
        batch_size = input_batches.size(1)

        BOS_token = self.basic_dict["<bos>"]
        EOS_token = self.basic_dict["<eos>"]
        PAD_token = self.basic_dict["<pad>"]

        # 初始化
        encoder_hidden = torch.zeros(self.enc_n_layers * self.enc_n_directions, batch_size, self.encoder.hid_dim,
                                     device=self.device)

        # encoder_output = [seq_len, batch, hid_dim * n directions]
        # encoder_hidden = [n_layers*n_directions, batch, hid_dim]
        encoder_output, encoder_hidden = self.encoder(
            input_batches, input_lengths, encoder_hidden)

        # 初始化
        decoder_input = torch.tensor([BOS_token] * batch_size, dtype=torch.long, device=self.device)
        if self.enc_n_directions == self.dec_n_directions:
            decoder_hidden = encoder_hidden
        else:
            L = encoder_hidden.size(0)
            decoder_hidden = encoder_hidden[range(0, L, 2)] + encoder_hidden[range(1, L, 2)]
        if self.predict:
            # 预测阶段使用
            # 一次只输入一句话
            assert batch_size == 1   #, "batch_size of predict phase must be 1!"
            output_tokens = []
            Nor_output = []
            Prf = 0.88
            Pri = 0.93
            while True:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # [1, 1]
                soft_max = softmax1(decoder_output.squeeze(0).data.numpy())
                max_value = max(soft_max)
                fuzz = random.uniform(0, 1)
                if fuzz > Prf and max_value > Pri:
                    topv, topi = decoder_output.topk(1, largest=False)
                    # print(topi)
                    # topi,_= topi.topk(1)
                else:
                    topv, topi = decoder_output.topk(1)

                # topv, topi = decoder_output.topk(1)
                Nor_output.append(torch.tensor(soft_max.index(max_value)))
                decoder_input = topi.squeeze(1)  # 上一个预测作为下一个输入
                output_token = topi.squeeze().detach().item()
                if output_token == EOS_token or len(output_tokens) == self.max_len:
                    break
                output_tokens.append(output_token)
            return output_tokens

        else:
            # 训练阶段
            max_target_length = max(target_lengths)
            all_decoder_outputs = torch.zeros((max_target_length, batch_size, self.decoder.output_dim),
                                              device=self.device)

            for t in range(max_target_length):
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    # decoder_output = [batch, output_dim]
                    # decoder_hidden = [n_layers*n_directions, batch, hid_dim]
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    all_decoder_outputs[t] = decoder_output
                    decoder_input = target_batches[t]  # 下一个输入来自训练数据
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    # [batch, 1]
                    topv, topi = decoder_output.topk(1)
                    all_decoder_outputs[t] = decoder_output
                    decoder_input = topi.squeeze(1)  # 下一个输入来自模型预测

            loss_fn = nn.NLLLoss(ignore_index=PAD_token)
            loss = loss_fn(
                all_decoder_outputs.reshape(-1, self.decoder.output_dim),  # [batch*seq_len, output_dim]
                target_batches.reshape(-1)  # [batch*seq_len]
            )
            return loss
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model,data_loader,optimizer,clip=1,teacher_forcing_ratio=0.5,print_every=None ): # None不打印
    model.predict = False
    model.train()

    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置
    start = time.time()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):

        # shape = [seq_len, batch]
        input_batchs = batch["src"]
        target_batchs = batch["trg"]
        # list
        input_lens = batch["src_len"]
        target_lens = batch["trg_len"]

        optimizer.zero_grad()

        loss = model(input_batchs, input_lens, target_batchs, target_lens, teacher_forcing_ratio)
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)
def evaluate( model,data_loader,print_every=None):
    model.predict = False
    model.eval()
    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置
    start = time.time()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            input_batchs = batch["src"]
            target_batchs = batch["trg"]
            # list
            input_lens = batch["src_len"]
            target_lens = batch["trg_len"]

            loss = model(input_batchs, input_lens, target_batchs, target_lens, teacher_forcing_ratio=0)
            print_loss_total += loss.item()
            epoch_loss += loss.item()
            if print_every and (i+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('\tCurrent Loss: %.4f' % print_loss_avg)
    return epoch_loss / len(data_loader)

def translate(model,sample,idx2token=None):
    model.predict = True
    model.eval()
    # shape = [seq_len, 1]
    input_batch = sample["src"]
    # list
    input_len = sample["src_len"]
    output_tokens = model(input_batch, input_len)
    #print(output_tokens)
    #output_tokens = [idx2token[t] for t in output_tokens]

    return output_tokens

INPUT_DIM = len(key)
OUTPUT_DIM = len(key)
# 超参数
BATCH_SIZE = 32
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LEARNING_RATE = 0.001 #学习率 0.0001
N_EPOCHS = 25
CLIP = 1

bidirectional = True
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, bidirectional)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, bidirectional)
model = Seq2Seq(enc, dec, device, basic_dict=basic_dict).to(device)

## encoder和encoder设置相同的学习策略
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_set = TranslationDataset(en_num_data, ch_num_data)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=padding_batch)
best_valid_loss = float('inf')

def train_text():
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, CLIP)
        # valid_loss = evaluate(model, train_loader)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        Tran_loss.append(train_loss)
        # Val_loss.append(valid_loss)
        Epoch.append(epoch + 1)
        torch.save(model, 'model/seq2seq_' + str(epoch+1) + '.pth')

# 加载最优权重
# model.load_state_dict(torch.load('model/seq2seq_50.pth'))

def makedata(word):
    data3=[]
    ma_data = ""
    for i in range(0, len(word), 2):
        ma_data=word[i] + word[i + 1]
        data3.append(key[ma_data])
    return data3
def dedata(words):
    data4=""
    for m in words:
        data4=data4 + idx2word[m]
    return data4
def transation(word):
    en_tokens = list(filter(lambda x: x != 0,word))  # 过滤零
    test_sample = {}
    test_sample["src"] = torch.tensor(en_tokens, dtype=torch.long, device=device).reshape(-1, 1)
    test_sample["src_len"] = [len(en_tokens)]

    data_1 = dedata(translate(model, test_sample, key)).replace("<unk>",'')
    data_2 = data_1.replace("<pad>", '')
    data_3 = data_2.replace("<bos>", '')
    return data_3
def draw():
    ftext = open('dataset.txt', 'a')
    plt.plot(Epoch, Tran_loss, color="red",marker='.',label="tran loss")
    plt.plot(Epoch, Val_loss, color="green", marker='*',label="val loss")
    plt.legend()
    plt.title("demo")
    plt.xlabel("epoch")
    plt.ylabel("tran/val loss")
    plt.show()
    for i in Epoch:
        ftext.write(str(i)+',')
    ftext.write('\n')
    ftext.write("Tran_loss:")
    for x in Tran_loss:
        ftext.write(str(x) + ',')
    ftext.write('\n')
    ftext.write("Val_loss:")
    for y in Val_loss:
        ftext.write(str(y) + ',')
    ftext.write('\n')
    ftext.close()
# draw()

def generator():
    try:
        Nor_num_data = 0
        Nor_unusual_data = 0
        Nor_num0 = 0
        b_time = time.time()
        # 连接从机地址,这里要注意端口号和IP与从机一致
        MASTER = modbus_tcp.TcpMaster(host="192.168.1.102", port=502)
        MASTER.set_timeout(0.5)
        LOGGER.info("connected")
        number = 0
        record_all_text = []
        record_right_text = []
        ftxt = open("modbus_tcp.txt", 'r')
        ftxt_recode = open("ModRSsim2_result_7.9.txt","w")
        fd = ftxt.readlines()
        while number <= 16001:
            number = number + 1
            uid = random.randint(0, 3999)
            length = random.randint(0, 3999)
            slave_fc = random.randint(0, 3999)

            str_uid = fd[uid][0:4]
            str_len = fd[length][8:12]
            str_slave = fd[slave_fc][12:16]
            str0 = str_uid + "0000" + str_len + str_slave

            text = transation(makedata(str0))
            modbus_tcp2 = str0 + text

            lenth = int(len(modbus_tcp2[12:]) / 2)
            length = hex(lenth).split("0x")
            length1 = length[1]
            for i in range(0, 4 - len(length[1])):
                length1 = "0" + length1
            string = list(modbus_tcp2)
            string[8:12] = length1
            modbus_tcp2 = "".join(string)

            Nor_num_data = Nor_num_data + 1
            Nor_flag, reasion = ch.check(modbus_tcp2)
            record_all_text.append(modbus_tcp2[14:16] + text)

            Nor_num = MASTER.execute1(modbus_tcp2[12:14], modbus_tcp2)

            if Nor_flag == False and Nor_num == 0:
                record_right_text.append(modbus_tcp2[14:16] + text)
                Nor_num0 = Nor_num0 + 1
                Nor_unusual_data = Nor_unusual_data + 1
            elif Nor_num == 0:
                record_right_text.append(modbus_tcp2[12:14] + text)
                Nor_num0 = Nor_num0 + 1
            target_numbers = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
            o_time = time.time()

            if Nor_num_data in target_numbers:
                epoch_mins, epoch_secs = epoch_time(b_time, o_time)
                new_record_text = list(set(record_all_text))
                new_right_text = list(set(record_right_text))
                record_len = len(new_record_text)
                right_len = len(new_right_text)
                epoch_mins, epoch_secs = epoch_time(b_time, o_time)
                print("###########################第",Nor_num_data,"测试用例################################")
                print("测试", Nor_num_data, "条数据", f'共用时: {epoch_mins}m {epoch_secs}s')
                print("      接收率：", format(Nor_num0 / Nor_num_data, '.4%'))
                print("      错误率：", format(1 - (Nor_num0 / Nor_num_data), '.4%'))
                print("非正常正确数据：", format((Nor_unusual_data / Nor_num_data), '.4%'))
                print("所有测试用例中不重复的测试用例个数：", record_len, format(record_len / Nor_num_data, '.4%'))
                print("正确测试用例中不重复的测试用例个数：", right_len, format(right_len / Nor_num0, '.4%'))
                ftxt_recode.write("共测试" + str(Nor_num_data) + "条数据," + str(f'共用时: {epoch_mins}m {epoch_secs}s') + "\n")
                ftxt_recode.write("      接收率：" + str(Nor_num0) + "  " + str(format(Nor_num0 / Nor_num_data,'.4%')) + "\n")
                ftxt_recode.write("      错误率：" + str(format(1 - (Nor_num0 / Nor_num_data), '.4%')) + "\n")
                ftxt_recode.write("非正常正确数据：" + str(Nor_unusual_data) + "  " + str(format((Nor_unusual_data / Nor_num_data), '.4%')) + "\n")
                ftxt_recode.write("所有测试用例中不重复的测试用例个数：" + str(record_len) + "  " + str(format(record_len / Nor_num_data, '.4%')) + "\n")
                ftxt_recode.write("正确测试用例中不重复的测试用例个数：" + str(right_len) + "  " + str(format(right_len / Nor_num0, '.4%')) + "\n")
        ftxt.close()
        ftxt_recode.close()
    except modbus_tk.modbus.ModbusError as err:
        LOGGER.error("%s- Code=%d" % (err, err.get_exception_code()))
if __name__ == '__main__':

    chose_num = int(input("1.最初模型训练; 2.测试用例生成"))
    if chose_num == 1:  # 模型生成
        train_text()
    elif chose_num == 2:  # 测试用例生成
        for i in range(1,2):
            model_name = 'model/seq2seq_' + str(9) + ".pth"
            print(model_name)
            model = torch.load(model_name)
            generator()

