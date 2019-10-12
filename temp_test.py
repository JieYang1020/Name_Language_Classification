from io import open
import glob
import unicodedata
import string
import os
import pandas as pd
from sklearn.utils import shuffle
from torchtext import data
from tqdm import tqdm
from torch.nn import init
from torchtext.vocab import GloVe, Vectors
from torchtext.data import Iterator, BucketIterator

#所有英文字母加上五个标点符号(包含一个空格)
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
# 将unicode转为ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
#build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []
# read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding = 'utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

file_path = r'C:\Users\lenovo\Desktop\Josie\自学\Pytorch_名字分类\data\data\names'
#数据整合成dataframe
total_data = pd.DataFrame(columns = ('content', 'category'))
for root, dirs, files in os.walk(file_path):
    for idx, file in enumerate(files):
        category = file.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(os.path.join(root, file))
        for line in lines:
            single_name = {'content':line, 'category':int(idx)}
            total_data = total_data.append(single_name, ignore_index = True)
#找到最长名字，做pad用(fix_length = 最长长度)
max_length = 0
for i in total_data['content']:
    if len(i) > max_length:
        max_length = len(i)
print('名字中最长长度为%d'%max_length)
#解决loss值不变的方法一：打乱数据集(total_data)
total_data = shuffle(total_data)
tokenize = lambda x: x.split()
#data.Field:定义样本的处理操作
TEXT = data.Field(sequential = True, tokenize = tokenize, lower = True, fix_length = 19)
LABEL = data.Field(sequential = False, use_vocab = False)
#将原始的corpus转换成data.Example实例(主要为data.Example.fromlist方法)
#都是train数据，就不用区分是train还是test了
def get_dataset(content_info, category_info, text_field, label_field):
    fields = [('content', text_field),
               ('category', label_field)]
    examples = []
    for text, label in zip(content_info, category_info):
        examples.append(data.Example.fromlist([text, label], fields))
    return examples, fields
train_examples, train_fields = get_dataset(total_data['content'], total_data['category'],
                                          TEXT, LABEL)
#使用torchtext.data.Dataset来构建数据集
train = data.Dataset(train_examples, train_fields)
TEXT.build_vocab(train)
weight_matrix = TEXT.vocab.vectors
#解决loss值不变的方法二：减小batch_size
train_iter = BucketIterator(train, batch_size = 50, device = -1,
                            sort_key = lambda x: len(x.content),
                            sort = False,sort_within_batch = False, repeat = False)
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR


class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.batch_size = 64
        self.hidden_size = 128
        self.vocab_size = len(TEXT.vocab)
        self.embedding_length = 300
        self.embedding_dropout = 0.2
        self.fc_dropout = 0.1
        # bidirectional没用上，会报错
        self.output_size = 18
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
        #         #指定预训练的词向量(即embedding层的权重)（Glove)
        weight_matrix = TEXT.vocab.vectors
        # self.word_embeddings.weight.data.copy_(weight_matrix)
        # 反向时不计算embeddin层的梯度(不更新embedding层的权重)，提升模型训练时间，对应优化器有需要调整的地方
        self.word_embeddings.weight.requires_grad = False
        #         self.word_embeddings.weight = nn.Parameter(word_embeddings, require_grad = True)
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        #         self.embed_dropout = nn.Dropout(self.embedding_dropout)
        #         self.fc_dropout = nn.Dropout(self.fc_dropout)
        #         self.relu = nn.ReLU()
        self.label = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_sentence):
        print("下面这行")
        print(input_sentence.shape)
        input = self.word_embeddings(input_sentence)
        print(input.shape)
        if input.shape[1] == self.batch_size:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(1, input.shape[1], self.hidden_size))
            c_0 = Variable(torch.zeros(1, input.shape[1], self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])
        #         final_output = self.relu(final_output)
        #         final_output = self.fc_dropout(final_output)
        #         print(final_output)
        return final_output


# 梯度裁剪来防止梯度爆炸
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        # nn.utils.clip_grad_norm(model.parameters(), 10)
        p.grad.data.clamp_(-clip_value, clip_value)


def main():
    model = LSTMClassifier()
    model.train()
    total_epoches_loss = 0
    total_epoches_acc = 0
    #     optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    # 动态监控lr,如多次(200)没有发生loss下降，则降低学习率
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # 传入优化器让学习器受其管理，当连续200次没有减少Loss时就减低lr(乘以0.9)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
    # pytorch中处理多分类用CrossEntropyLoss时，标签需从0开始
    loss_function = nn.CrossEntropyLoss()
    epoches = 5
    Loss_list = []
    Accuracy_list = []
    for i in range(epoches):
        each_batch_loss = 0
        each_batch_acc = 0
        for epoch, batch in enumerate(train_iter):
            optimizer.zero_grad()
            predicted = model(batch.content)
            loss = loss_function(predicted, batch.category)
            num_corrects = (torch.max(predicted, 1)[1].view(batch.category.size()).data
                            == batch.category.data).float().sum()
            acc = 100.0 * num_corrects / len(batch)
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 10)
            clip_gradient(model, 1e-1)
            optimizer.step()
            scheduler.step()
            each_batch_loss += loss.item()
            each_batch_acc += acc.item()
            Loss_list.append(each_batch_loss)
            Accuracy_list.append(each_batch_acc)
        total_epoches_loss += each_batch_loss
        total_epoches_acc += each_batch_acc
        print('第%d个epoch的loss值为%f' % (i + 1, (each_batch_loss / len(train_iter))))
        print('第%d个epoch的准确率为%f' % (i + 1, (each_batch_acc / len(train_iter) / 100.0)))


if __name__ == '__main__':
    main()
