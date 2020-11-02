# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score

class MLP(nn.Module):
	def __init__(self, maxlen, drop_rate=0.5):
		super(MLP, self).__init__()
		self.maxlen = maxlen
		# Define your layers here
		self.layers = nn.Sequential(
			nn.Linear(268,128),
			nn.ReLU(inplace=True),
			# nn.Dropout(0.2),
			nn.Linear(128,2)
			# nn.Conv1d(64,64,3,padding=1),
			# nn.BatchNorm1d(64),
			# nn.ReLU(inplace=True),
			# nn.Dropout(0.2),
			# nn.MaxPool1d(3,stride=3),
		)
		# self.linear = nn.Linear(2688,2)
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# the 2-class prediction output is named as "logits"
		# inp = torch.tensor(x, dtype=torch.float32)
		x = x.float()
		if len(x.shape) == 1:
			x = x.unsqueeze(0)
		# print(x.shape)
		logits = self.layers(x)
		# print(logits.shape)
		# logits = logits.reshape(100,-1)

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return torch.softmax(logits,dim=1)
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

