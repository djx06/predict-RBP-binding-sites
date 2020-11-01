# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score

class CNN(nn.Module):
	def __init__(self, maxlen, drop_rate=0.5):
		super(CNN, self).__init__()
		self.maxlen = maxlen
		# Define your layers here
		self.layers = nn.Sequential(
			nn.Conv1d(maxlen,128,3,padding=1),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			nn.MaxPool1d(3,stride=3),
			# nn.Conv1d(64,64,3,padding=1),
			# nn.BatchNorm1d(64),
			# nn.ReLU(inplace=True),
			# nn.Dropout(0.2),
			# nn.MaxPool1d(3,stride=3),
		)
		self.linear = nn.Linear(2688,2)
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# the 2-class prediction output is named as "logits"
		# inp = torch.tensor(x, dtype=torch.float32)
		x = x.float()
		# print(x.shape)
		logits = self.layers(x)
		# print(x.shape)
		logits = logits.reshape(100,-1)
		logits = self.linear(logits)

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		# print(pred)
		# print(y)
		# labels = torch.tensor(y,dtype=torch.int64)
		y = y.long()
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch
		# print(acc)
		# auc = roc_auc_score(pred.int(), y.int())
		auc = 0
		return loss, acc,auc


class LSTM(nn.Module):
	def __init__(self, maxlen, drop_rate=0.5):
		super(LSTM, self).__init__()
		# Define your layers here
		self.dropout = drop_rate
		self.maxlen = maxlen
		self.rnn = nn.LSTM(input_size=maxlen, hidden_size=300, num_layers=1, batch_first=True)
		self.linear = nn.Linear(269,2,bias=True)
		self.h0 = torch.zeros(1,100,300)
		self.c0 = torch.zeros(1,100,300)
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# the 2-class prediction output is named as "logits"
		x = x.float()
		print(x.shape)
		output,(hn,cn) = self.rnn(x,(self.h0,self.c0))
		print(output.shape)
		# logits = logits.reshape(100,-1)
		logits = self.linear(output)

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc

# class LSTM(nn.Module):
#     def __init__(self,parameter):
#         super(LSTM, self).__init__()
#         self.parameter = parameter
#         self.sen_input = None
#         self.dropout_rate = 0.5
#         self.labels = None
#         self.embedded_sentens = nn.Embedding(self.parameter.num_char,self.parameter.char_dim)
#         self.rnn = nn.LSTM(input_size=300, hidden_size=300, num_layers=1, batch_first=True)
#         self.linear = nn.Linear(300,8,bias=True)
#         self.h0 = torch.zeros(1,64,300)
#         self.c0 = torch.zeros(1,64,300)
#         self.loss = nn.CrossEntropyLoss()
#     def forward(self):
#         sen_in = self.embedded_sentens(self.sen_input)
#         print(sen_in.size())
#         output, (hn, cn) = self.rnn(sen_in, (self.h0, self.c0))
#         output = self.linear(output)
#         # print(output.size())
#         # output = output.permute(1,0,2)
#         output = output[:,399,:]
#         loss = self.loss(output,self.labels)
#         _, maxes = torch.max(output,dim=1)
#         return loss, maxes