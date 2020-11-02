# -*- coding: utf-8 -*-
import sys
import argparse
import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import MLP
# from load_data import load_data
from keras.utils.np_utils import to_categorical

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100,
	help='Batch size for mini-batch training and evaluating. Default: 100')
parser.add_argument('--num_epochs', type=int, default=50,
	help='Number of training epoch. Default: 20')
parser.add_argument('--learning_rate', type=float, default=1e-3,
	help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--drop_rate', type=float, default=0.5,
	help='Drop rate of the Dropout Layer. Default: 0.5')
parser.add_argument('--is_test',action="store_true",
	help='True to train and False to inference. Default: True')
parser.add_argument('--inference_version', type=int, default=0,
	help='The version for inference. Set 0 to use latest checkpoint. Default: 0')
parser.add_argument('--pathname',type=str,default='./data',
	help='Pathname of training data')
parser.add_argument('--dataset',type=str,default='./PARCLIP_MOV10_Sievers.train',
	help='Name of dataset')	
parser.add_argument('--model',type=str,default='mlp',
	help='Choose an ML Model')	
parser.add_argument('--name',type=str,default='mlp',
	help='Name of this model')	
parser.add_argument('--data_dir',type=str,default='./data/PARCLIP_MOV10_Sievers.ls.positives.txt',
	help='Test path')	
parser.add_argument('--train_dir',type=str,default='./train',
	help='Train Direction')
parser.add_argument('--savepred',type=str,default='predict.txt',
	help='Prediction')
args = parser.parse_args()


bases = ['A', 'C', 'G', 'U']
# base_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
base_dict = {}
bases_len = len(bases)
max_len = 376
bags = []
def convert_to_index(str,word_len):
   '''
   convert a sequence 'str' of length 'word_len' into index in 0~4^(word_len)-1
   '''
   output_index = 0
   for i in range(word_len):
      output_index = output_index * bases_len + base_dict[str[i]]
   return output_index

def shuffle(X, y, shuffle_parts):
	chunk_size = int(len(X) / shuffle_parts)
	shuffled_range = list(range(chunk_size))

	X_buffer = np.copy(X[0:chunk_size])
	y_buffer = np.copy(y[0:chunk_size])

	for k in range(shuffle_parts):
		np.random.shuffle(shuffled_range)
		for i in range(chunk_size):
			X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
			y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

		X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
		y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

	return X, y


# def shuffle(trainX,trainY):
#    '''
#    random shuffle the training data
#    '''
#    np.random.seed(0)
#    train=np.hstack([trainY[:,None],trainX]).astype('float32')
#    train=np.random.permutation(train)
#    train=train[:120000,:]
#    trainX=train[:,1:].astype('float32')
#    trainY=train[:,0].astype('float32')
#    return trainX,trainY

def predicting(filename,model,savepred=None):
   '''
   predict for sequences in 'filename' using the preprocessing transform 'scaler' and the trained model '_model'
   '''
   print ("Predicting",filename)
   start=time.time()
   if savepred is not None:
      fout=open(savepred,'w')
   try:
      for line in open(filename):
         if line[0]=='>':
            if savepred is not None:
               fout.write(line)
            continue
         elif ('n' in line or 'N' in line):
            if savepred is not None:
               fout.write('Error!\n')
         else:
            line=line.strip('\n').strip('\r')
            testX=np.array(extract_features(line))[None,:]
            pred = model.predict_proba(testX)[:,1]
            if savepred is not None:
               fout.write('%f\n'%float(pred[0]))
   finally:
      if savepred is not None:
         fout.close()

def leave_out(trainX,trainY):
   '''
   split the data into 70% for training and 30% for testing
   '''
   m=int(0.7*len(trainX))
   testX=trainX[m:,:]
   testY=trainY[m:]
   trainX=trainX[:m,:]
   trainY=trainY[:m]
   return trainX,trainY,testX,testY

def extract_features(line):
   '''
   extract features from a sequence of RNA
   
   To do: alternative ways to generate features can be used. 
   '''
   core_seq = line
   for i in 'agctu\n':
      core_seq = core_seq.replace(i, '')
   core_seq = core_seq.replace('T','U')
   core_seq = core_seq.replace('N','')
   final_output=[]
   for word_len in [1,2,3]:
      output_count_list = [0 for i in range(bases_len ** word_len)]
      for i in range(len(core_seq)-word_len+1):
         output_count_list[convert_to_index(core_seq[i:i+word_len],word_len)] +=1
      final_output.extend(output_count_list)
   return final_output

def get_bags(s,k,base):
    if len(s) == k-1:
        for i in base:
            bags.append(s+i)
        return
    else:
        for i in base:
            next = s + i
            get_bags(next,k,base)
    return

def get_kmers(sequence,k):
    begin = 0
    end = begin+k
    k_list = []
    while end <= len(sequence):
        k_list.append(sequence[begin:end])
        begin+=1
        end+=1
    return k_list

def one_hot_features(line):
	''' 
	extract one_hot features from a sequence of RNA
	'''
	global max_len
	# if len(line) > max_len:
	# 	max_len = len(line)
	core_seq = line
	for i in 'agctu\n':
		core_seq = core_seq.replace(i, '')
	core_seq = core_seq.replace('T','U')
	core_seq = core_seq.replace('N','')
	# line = core_seq
	line = line.upper()
	line = line.replace('T','U')
	line = line.replace('N','')
	indexes = []
	k_mers = get_kmers(line,3)
	for i in k_mers:
		indexes.append(base_dict[i])
	indexes = np.array(indexes)
	one_hot = to_categorical(indexes, num_classes=None)
	# for i in range(150,one_hot.shape[0]-150):
	# 	one_hot[i] = one_hot[i] * 5
		# print(one_hot[i])
	if one_hot.shape[1] != 64:
		add = np.zeros((one_hot.shape[0],64-one_hot.shape[1]))
		one_hot = np.append(one_hot,add,axis=1)
	if one_hot.shape[0] < max_len:
		padding = np.zeros((max_len-one_hot.shape[0],64))
		one_hot = np.append(one_hot,padding,axis=0)
	# print(one_hot.shape)
	# print(line)
	# print(one_hot)
	return one_hot



def load_data(filename,check=False,savecheck='check'):
	'''
	use the extract_features function to extract features for all sequences in the file specified by 'filename'
	'''
	print('Processing ',filename)
	start=time.time()
	total_output=[]
	valid=[]
	f = open(filename,'r')
	data = json.load(f)
	for line in data:
		valid.append(1)
		total_output.append(data[line])
	output_arr=np.array(total_output)
	if (check):
		np.save(savecheck,np.array(valid))
	global max_len
	print(max_len)
	print(output_arr.shape)
	end=time.time()
	print ('Finished loading in',end-start,'s\n')
	return output_arr


def train_epoch(model, X, y, optimizer): # Training Process
	model.train()
	loss, acc, roc = 0.0, 0.0, 0.0
	st, ed, times = 0, args.batch_size, 0
	while st < len(X) and ed <= len(X):
		optimizer.zero_grad()
		X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
		loss_, acc_, roc_ = model(X_batch, y_batch)

		loss_.backward()
		optimizer.step()

		loss += loss_.cpu().data.numpy()
		acc += acc_.cpu().data.numpy()
		roc += roc_
		st, ed = ed, ed + args.batch_size
		times += 1
	loss /= times
	acc /= times
	roc /= times
	return acc, loss, roc


def valid_epoch(model, X, y): # Valid Process
	model.eval()
	loss, acc, roc = 0.0, 0.0, 0.0
	st, ed, times = 0, args.batch_size, 0
	while st < len(X) and ed <= len(X):
		X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
		loss_, acc_, roc_= model(X_batch, y_batch)

		loss += loss_.cpu().data.numpy()
		acc += acc_.cpu().data.numpy()
		roc += roc_
		st, ed = ed, ed + args.batch_size
		times += 1
	loss /= times
	acc /= times
	roc /= times
	return acc, loss, roc


def inference(model, X): # Test Process
	model.eval()
	pred_ = model(torch.from_numpy(X).to(device))
	return pred_.cpu().data.numpy()


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cpu")
	# print(device)
	if not os.path.exists(args.train_dir):
		os.mkdir(args.train_dir)
	get_bags("",3,bases)
	print(bags)
	count = 0
	for key in bags:
		base_dict[key] = count
		count+=1
	# print(args.is_train)
	if not args.is_test:

		pos_filename=os.path.join(args.pathname,args.dataset+'.positives.txt')
		neg_filename=os.path.join(args.pathname,args.dataset+'.negatives.txt')
		pos_trainX=load_data(pos_filename)
		pos_trainY=np.ones(len(pos_trainX))
		neg_trainX=load_data(neg_filename)
		neg_trainY=np.zeros(len(neg_trainX))

		trainX=np.vstack([pos_trainX,neg_trainX])
		trainY=np.hstack([pos_trainY,neg_trainY])
		trainX,trainY=shuffle(trainX,trainY,1)
		X_train,y_train,X_val,y_val=leave_out(trainX,trainY)

		model = MLP(max_len,drop_rate=args.drop_rate)
		model.to(device)
		# f = open("../result/cnn_nobn.txt",'w')
		# f.write("TrainAcc\tTrainLoss\tValAcc\tValLoss\n")
		optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

		# model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
		# if os.path.exists(model_path):
		# 	cnn_model = torch.load(model_path)

		pre_losses = [1e18] * 3
		best_val_acc = 0.0
		for epoch in range(1, args.num_epochs+1):
			start_time = time.time()
			train_acc, train_loss,train_roc = train_epoch(model, X_train, y_train, optimizer)
			X_train, y_train = shuffle(X_train, y_train, 1)

			val_acc, val_loss, val_roc = valid_epoch(model, X_val, y_val)

			if val_acc >= best_val_acc:
				best_val_acc = val_acc
				best_epoch = epoch
				# test_acc, test_loss = valid_epoch(model, X_test, y_test)
				with open(os.path.join(args.train_dir, 'checkpoint_{}.pth.tar'.format(args.name)), 'wb') as fout:
					torch.save(model, fout)

			epoch_time = time.time() - start_time
			print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
			print("  learning rate:                 " + str(optimizer.param_groups[0]['lr']))
			print("  training loss:                 " + str(train_loss))
			print("  training accuracy:             " + str(train_acc))
			print("  training roc:                  " + str(train_roc))
			print("  validation loss:               " + str(val_loss))
			print("  validation accuracy:           " + str(val_acc))
			print("  validation roc:                " + str(val_roc))
			print("  best epoch:                    " + str(best_epoch))
			print("  best validation accuracy:      " + str(best_val_acc))
			# print("  test loss:                     " + str(test_loss))
			# print("  test accuracy:                 " + str(test_acc))

			if train_loss > max(pre_losses):
				for param_group in optimizer.param_groups:
					param_group['lr'] = param_group['lr'] * 0.9995
			pre_losses = pre_losses[1:] + [train_loss]

	else:
		print ("Predicting",args.data_dir)
		model = MLP(max_len,drop_rate=0.5)
		model.to(device)
		model_path = os.path.join(args.train_dir, 'checkpoint_%s.pth.tar' % args.name)
		if os.path.exists(model_path):
			model = torch.load(model_path)
		start=time.time()
		
		f = open(args.data_dir,'r')
		data = json.load(f)
		# for line in data:
		# 	valid.append(1)
		# 	total_output.append(data[line])
		# output_arr=np.array(total_output)

		if args.savepred is not None:
			fout=open(args.savepred,'w')
		try:
			for line in data:
				fout.write(line+'\n')
				testX = np.array(data[line])
				pred = inference(model, testX)[0]
				# print(pred)
				if args.savepred is not None:
					fout.write('%f\n'%float(pred[1]))
		finally:
			if args.savepred is not None:
				fout.close()

		# model.to(device)
		# model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
		# if os.path.exists(model_path):
		# 	model = torch.load(model_path)

		# X_train, X_test, y_train, y_test = load_data(args.data_dir)

		# count = 0
		# for i in range(len(X_test)):
		# 	test_image = X_test[i].reshape((1, 3, 32, 32))
		# 	result = inference(model, test_image)[0]
		# 	if result == y_test[i]:
		# 		count += 1
		# print("test accuracy: {}".format(float(count) / len(X_test)))
