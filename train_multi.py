import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import argparse
import time
from tqdm import tqdm, trange
from multimodal import FSSAM,SAM,OneCNN,DatanetMLP,DatanetCNN,BiLSTM,DeepPacket,TSCRNN,Multitask_multimodal,Multimodal_MMoE,MMoE,WF
from split_train import websites_kind, cdn_kind, load_epoch_data, max_byte_len
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import pickle
import numpy as np

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

lr = 1.0e-4
packet_number = 10

class Dataset(torch.utils.data.Dataset):
	"""docstring for Dataset"""
	#def __init__(self, x, y, label):
	def __init__(self,label1,label2,direction_length,time, x,y):
		super(Dataset, self).__init__()
		'''
		self.x = x
		self.y = y
		self.label1 = label1
		self.label2 = label2
		'''

		self.label1 = label1
		self.label2 = label2
		self.direction_length = direction_length
		self.time = time
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		#origin 784  now about packet*length
		#1DCNN
		#traffic, target = self.x[idx],int(self.label1[idx])
		#traffic, target = self.x[idx],int(self.label2[idx])
		#traffic=traffic.reshape(1,1,-1)

		 
		#DatasetMLP
		#traffic, target = self.x[idx], int(self.label1[idx])
		#traffic, target = self.x[idx], int(self.label2[idx])
		
		
		#DatasetCNN
		#traffic, target = self.x[idx], int(self.label1[idx])
		#traffic, target = self.x[idx], int(self.label2[idx])
		#traffic = traffic.reshape(1,packet_number,-1)
		
		#BiLSTM TSCRNN DeepPacket
		'''
		traffic, target = self.x[idx], int(self.label1[idx])
		traffic, target = self.x[idx], int(self.label2[idx])
		traffic = traffic.reshape(1,-1)
		
		return traffic, target
		'''
		#traffic = self.x[idx]
		#traffic = traffic.reshape(1,1,-1)

		return self.label1[idx], self.label2[idx], self.direction_length[idx], self.time[idx],self.x[idx], self.y[idx]

def paired_collate_fn(insts):
	label1, label2, direction_length, time, x, y = list(zip(*insts))
	return torch.LongTensor(label1), torch.LongTensor(label2),torch.LongTensor(direction_length), torch.LongTensor(time), torch.LongTensor(x), torch.LongTensor(y)

def cal_loss(pred, gold, cls_ratio=None):
	gold = gold.contiguous().view(-1)
	# By default, the losses are averaged over each loss element in the batch. 
	loss = F.cross_entropy(pred, gold)

	# torch.max(a,0) 返回每一列中最大值的那个元素，且返回索引
	pred = F.softmax(pred, dim = -1).max(1)[1]
	# 相等位置输出1，否则0
	#print('========================')
	#print(pred)
	#print(gold)
	n_correct = pred.eq(gold)
	acc = n_correct.sum().item() / n_correct.shape[0]

	return loss, acc*100


def cal_multi_loss(pred1, gold1, pred2, gold2, cls_ratio=None):
	loss1, acc1 = cal_loss(pred1,gold1)
	loss2, acc2 = cal_loss(pred2,gold2)
	#print(loss1,loss2)
	#loss = loss1/(loss1+loss2)*loss1+loss2/(loss2+loss2)*loss2
	loss = 3*loss1+2*loss2
	return loss, acc1, acc2

def test_epoch(model, test_data):
	''' Epoch operation in training phase'''
	model.eval()

	total_acc1 = []
	total_acc2 = []
	total_pred1 = []
	total_pred2 = []
	total_score = []
	total_time = []
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		test_data, mininterval=2,
		desc='  - (Testing)   ', leave=False):
		
		# prepare data
		#src_seq, src_seq2, gold = batch
		gold1, gold2, direction, time1, src_seq, src_seq2 = batch
		#src_seq, src_seq2, gold = src_seq.cuda(), src_seq2.cuda(), gold.cuda()
		gold1, gold2, direction, time1, src_seq, src_seq2 = gold1.cuda(), gold2.cuda(), direction.cuda(), time1.cuda(), src_seq.cuda(), src_seq2.cuda()
		
		gold1 = gold1.contiguous().view(-1)
		gold2 = gold2.contiguous().view(-1)

		# forward
		torch.cuda.synchronize()
		start = time.time()
		#pred1, pred2, score = model(direction, time1, src_seq, src_seq2)
		pred1, pred2 = model(direction, time1, src_seq, src_seq2)
		#pred1, pred2 = model(direction.to(torch.float32),time1.to(torch.float32),src_seq.to(torch.float32),src_seq2.to(torch.float32))
		torch.cuda.synchronize()
		end = time.time()
		# 相等位置输出1，否则0
		n_correct1 = pred1.eq(gold1)
		acc1 = n_correct1.sum().item()*100 / n_correct1.shape[0]
		total_acc1.append(acc1)
		total_pred1.extend(pred1.long().tolist())

		n_correct2 = pred2.eq(gold2)
		acc2 = n_correct2.sum().item()*100 / n_correct2.shape[0]
		total_acc2.append(acc2)
		total_pred2.extend(pred2.long().tolist())

		#total_score.append(torch.mean(score, dim=0).tolist())
		total_time.append(end - start)
	'''
	return sum(total_acc1)/len(total_acc1), sum(total_acc2)/len(total_acc2), np.array(total_score).mean(axis=0), \
	total_pred1, total_pred2, sum(total_time)/len(total_time)
	'''
	return sum(total_acc1)/len(total_acc1), sum(total_acc2)/len(total_acc2), \
	total_pred1, total_pred2, sum(total_time)/len(total_time)

def test_basic_epoch(model, test_data):
	''' Epoch operation in training phase'''
	model.eval()

	total_acc = []
	total_pred = []
	total_time = []
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		test_data, mininterval=2,
		desc='  - (Testing)   ', leave=False):
		
		# prepare data
		#src_seq,  gold = batch
		#src_seq,  gold = src_seq.cuda(), gold.cuda()
		gold1, gold2, direction, time1, src_seq, src_seq2 = batch
		gold1, gold2, direction, time1, src_seq, src_seq2 = gold1.cuda(), gold2.cuda(), direction.cuda(), time1.cuda(), src_seq.cuda(), src_seq2.cuda()
		#gold = gold.contiguous().view(-1)
		gold1 = gold1.contiguous().view(-1)
		gold2 = gold1.contiguous().view(-1)

		# forward
		torch.cuda.synchronize()
		start = time.time()
		#pred= model(direction.to(torch.float32),time1.to(torch.float32),src_seq.to(torch.float32),src_seq2.to(torch.float32))
		pred= model(direction,time1,src_seq,src_seq2)
		torch.cuda.synchronize()
		end = time.time()
		# 相等位置输出1，否则0
		n_correct = pred.eq(gold2)
		acc = n_correct.sum().item()*100 / n_correct.shape[0]
		total_acc.append(acc)
		total_pred.extend(pred.long().tolist())
		total_time.append(end - start)

	return sum(total_acc)/len(total_acc),  \
	total_pred, sum(total_time)/len(total_time)

def train_epoch(model, training_data, optimizer):
	''' Epoch operation in training phase'''
	model.train()

	total_loss = []
	total_acc1 = []
	total_acc2 = []
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		training_data, mininterval=2,
		desc='  - (Training)   ', leave=False):

		# prepare data
		#src_seq, src_seq2, gold = batch
		gold1, gold2, direction, time1, src_seq, src_seq2 = batch
		#src_seq, src_seq2, gold = src_seq.cuda(), src_seq2.cuda(), gold.cuda()
		gold1, gold2, direction, time1, src_seq, src_seq2 = gold1.cuda(), gold2.cuda(), direction.cuda(), time1.cuda(), src_seq.cuda(), src_seq2.cuda()

		optimizer.zero_grad()
		# forward
		#pred1,pred2 = model(direction.to(torch.float32),time1.to(torch.float32),src_seq.to(torch.float32),src_seq2.to(torch.float32))
		pred1,pred2 = model(direction,time1,src_seq,src_seq2)
		#loss_per_batch, acc_per_batch = cal_loss(pred, gold)
		loss_per_batch, acc1_per_batch, acc2_per_batch = cal_multi_loss(pred1, gold1, pred2, gold2)
		# update parameters
		loss_per_batch.backward()
		optimizer.step()

		# 只有一个元素，可以用item取而不管维度
		total_loss.append(loss_per_batch.item())
		total_acc1.append(acc1_per_batch)
		total_acc2.append(acc2_per_batch)

	return sum(total_loss)/len(total_loss), sum(total_acc1)/len(total_acc1), sum(total_acc2)/len(total_acc2)

def train_basic_epoch(model, training_data, optimizer):
	''' Epoch operation in training phase'''
	model.train()

	total_loss = []
	total_acc = []
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		training_data, mininterval=2,
		desc='  - (Training)   ', leave=False):

		# prepare data
		#src_seq, gold = batch
		gold1, gold2, direction, time1, src_seq, src_seq2 = batch
		#src_seq, gold = src_seq.cuda(), gold.cuda()
		gold1, gold2, direction, time1, src_seq, src_seq2 = gold1.cuda(), gold2.cuda(), direction.cuda(), time1.cuda(), src_seq.cuda(), src_seq2.cuda()

		optimizer.zero_grad()
		# forward
		#pred = model(direction.to(torch.float32),time1.to(torch.float32),src_seq.to(torch.float32),src_seq2.to(torch.float32))
		pred = model(direction,time1,src_seq,src_seq2)
		#loss_per_batch, acc_per_batch = cal_loss(pred, gold)
		loss_per_batch, acc_per_batch = cal_loss(pred, gold2)
		# update parameters
		loss_per_batch.backward()
		optimizer.step()

		# 只有一个元素，可以用item取而不管维度
		total_loss.append(loss_per_batch.item())
		total_acc.append(acc_per_batch)

	return sum(total_loss)/len(total_loss), sum(total_acc)/len(total_acc)

def main(i, flow_dict):
	f = open('results/results_%d.txt'%i, 'w')
	f.write('Train Loss Time Test\n')
	print('Train Loss Time Test')
	f.flush()

	#model = SAM(num_class=len(cdn_kind), max_byte_len=max_byte_len).cuda()
	model = Multitask_multimodal(num_class1=len(websites_kind), num_class2=len(cdn_kind), max_byte_len=max_byte_len).cuda()
	#model = FSSAM(num_class=len(websites_kind), max_byte_len=max_byte_len).cuda()

	#model = MMoE(num_class1=len(websites_kind), num_class2=len(cdn_kind),feature_dim=packet_number*(max_byte_len+2),expert_dim=64,n_expert=2,n_task=2,use_gate=True).cuda()
	#model = Multimodal_MMoE(num_class1=len(websites_kind), num_class2=len(cdn_kind),max_byte_len=max_byte_len,expert_dim=64,n_expert=3,n_task=2,use_gate=True).cuda()
	#model = WF(len(cdn_kind)).cuda()
	#model = OneCNN(len(cdn_kind)).cuda()
	#model = DatanetMLP(len(websites_kind)).cuda()
	#model = DatanetCNN(len(websites_kind)).cuda()
	#model = BiLSTM(len(websites_kind)).cuda()
	#model = DeepPacket(len(cdn_kind)).cuda()
	#model = TSCRNN(len(cdn_kind)).cuda()
	nParams = sum([p.nelement() for p in model.parameters()])
	print('* number of parameters: %d' % nParams)
	optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
	#optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # for basic model
	loss_list = []
	best_acc = 0
	best_acc1 = 0
	best_acc2 = 0
	for epoch_i in trange(15, mininterval=2, \
		desc='  - (Training Epochs)   ', leave=False):
		print('now is training: ' , epoch_i)
		print ('Loading training data....')
		label1,label2,direction_length,time,train_x,train_y = load_epoch_data(flow_dict, 'train')
		
		#for x y label
		training_data = torch.utils.data.DataLoader(
				Dataset(label1=label1,
					label2=label2,
					direction_length=direction_length,
					#direction_length=torch.zeros(packet_number,1),
					time=time,
					#time = torch.zeros(packet_number,1),
					x=train_x, y=train_y),
				num_workers=0,
				collate_fn=paired_collate_fn,
				batch_size=128,
				shuffle=True
			)
		
		'''
		#for x label
		training_data = torch.utils.data.DataLoader(
				Dataset(x=train_x, label1=label1, label2 = label2),
				num_workers=0,
				batch_size=128,
				shuffle=True
			)
		'''
		train_loss, train_acc1, train_acc2 = train_epoch(model, training_data, optimizer)
		#train_loss, train_acc = train_basic_epoch(model, training_data, optimizer)

		print ('Loading testing data....')
		label1,label2,direction_length,time,test_x,test_y = load_epoch_data(flow_dict, 'test')
		
		#for x y label
		test_data = torch.utils.data.DataLoader(
				Dataset(label1=label1,
					label2=label2,
					direction_length=direction_length,
					#direction_length=torch.zeros(packet_number,1),
					time=time,
					#time = torch.zeros(packet_number,1),
					x=test_x, y=test_y),
				num_workers=0,
				collate_fn=paired_collate_fn,
				batch_size=128,
				shuffle=False
			)
		'''
		#for x label
		test_data = torch.utils.data.DataLoader(
				Dataset(x=test_x, label1=label1, label2 = label2),
				num_workers=0,
				batch_size=128,
				shuffle=False
			)
		'''
		#test_acc1, test_acc2 , score, pred1, pred2, test_time = test_epoch(model, test_data)
		test_acc1, test_acc2 , pred1, pred2, test_time = test_epoch(model, test_data)
		#test_acc, pred, test_time = test_basic_epoch(model, test_data)

		#test_acc = accuracy_score(label2, pred)
		test_acc1 = accuracy_score(label1, pred1)
		test_acc2 = accuracy_score(label2, pred2)

		with open('results/atten_%d.txt'%i, 'w') as f2:
			#f2.write(' '.join(map('{:.4f}'.format, score)))
			f2.write('1'+'\n')

		'''
		#Single task learning
		# write F1, PRECISION, RECALL
		with open('results/metric_%d.txt'%i, 'w') as f3:
			f3.write(str(label2.reshape(1,-1).tolist()))
			f3.write(str(pred))
			f3.write('F1 PRE REC\n')
			p, r, fscore, _ = precision_recall_fscore_support(label2, pred)
			print('accuracy is : ',test_acc)
			for a, b, c in zip(fscore, p, r):
				# for every cls
				f3.write('%.2f %.2f %.2f\n'%(a, b, c))
				f3.flush()

		# write Confusion Matrix
		with open('results/cm_%d.pkl'%i, 'wb') as f4:
			pickle.dump(confusion_matrix(label2, pred, normalize='true'), f4)

		# write ACC
		f.write('%.2f %.4f %.6f %.2f\n'%(train_acc, train_loss, test_time, test_acc))
		print (train_acc, train_loss, test_time, test_acc)
		if train_acc>99:
			break
		f.flush()
		'''
		
		
		#Multitask learning
		# write F1, PRECISION, RECALL
		now_acc = test_acc1 + test_acc2
		print (train_acc1, train_acc2, train_loss, test_time, test_acc1, test_acc2)
		'''
		if test_acc1 > best_acc1:
			best_acc1 = test_acc1
			with open('results/metric1_%d.txt'%i, 'w') as f3:
				f3.write(str(label1.reshape(1,-1).tolist())+'\n')
				f3.write(str(pred1)+'\n')
				f3.write('F1 PRE REC\n')
				p, r, fscore, _ = precision_recall_fscore_support(label1, pred1)
				print('accuracy1 is : ',test_acc1)
				for a, b, c in zip(fscore, p, r):
					# for every cls
					f3.write('%.2f %.2f %.2f\n'%(a, b, c))
					f3.flush()
				if len(fscore) != len(websites_kind):
					a = set(pred1)
					b = set(label1[:,0])
					f3.write('%s\n%s'%(str(a), str(b)))
		if test_acc2 > best_acc2:
			best_acc2 = test_acc2
			with open('results/metric2_%d.txt'%i, 'w') as f3:
				f3.write(str(label2.reshape(1,-1).tolist())+'\n')
				f3.write(str(pred2)+'\n')
				f3.write('F1 PRE REC\n')
				p, r, fscore, _ = precision_recall_fscore_support(label2, pred2)
				print('accuracy2 is : ',test_acc2)
				for a, b, c in zip(fscore, p, r):
					# for every cls
					f3.write('%.2f %.2f %.2f\n'%(a, b, c))
					f3.flush()
				if len(fscore) != len(cdn_kind):
					a = set(pred2)
					b = set(label2[:,0])
					f3.write('%s\n%s'%(str(a), str(b)))
		'''

		with open('results/metric_%d.txt'%i, 'w') as f3:
			f3.write(str(label1.reshape(1,-1).tolist())+'\n')
			f3.write(str(pred1)+'\n')
			f3.write('F1 PRE REC\n')
			p, r, fscore, _ = precision_recall_fscore_support(label1, pred1)
			print('accuracy1 is : ',test_acc1)
			for a, b, c in zip(fscore, p, r):
				# for every cls
				f3.write('%.2f %.2f %.2f\n'%(a, b, c))
				f3.flush()
			if len(fscore) != len(websites_kind):
				a = set(pred1)
				b = set(label1[:,0])
				f3.write('%s\n%s'%(str(a), str(b)))

			f3.write('=====================Now is 2================='+'\n')
			f3.write(str(label2.reshape(1,-1).tolist())+'\n')
			f3.write(str(pred2)+'\n')
			f3.write('F1 PRE REC\n')
			p, r, fscore, _ = precision_recall_fscore_support(label2, pred2)
			print('accuracy2 is : ',test_acc2)
			for a, b, c in zip(fscore, p, r):
				# for every cls
				f3.write('%.2f %.2f %.2f\n'%(a, b, c))
				f3.flush()
			if len(fscore) != len(cdn_kind):
				a = set(pred2)
				b = set(label2[:,0])
				f3.write('%s\n%s'%(str(a), str(b)))


		# write Confusion Matrix
		with open('results/cm_1_%d.pkl'%i, 'wb') as f4:
			pickle.dump(confusion_matrix(label1, pred1, normalize='true'), f4)
		with open('results/cm_2_%d.pkl'%i, 'wb') as f5:
			pickle.dump(confusion_matrix(label2, pred2, normalize='true'), f5)

		# write ACC
		f.write('%.4f %.4f %.4f %.6f %.4f %.4f\n'%(train_acc1, train_acc2, train_loss, test_time, test_acc1, test_acc2))
		#if train_acc1>90 or train_acc2>90:
		#	break
		f.flush()
		
		

		# # early stop
		# if len(loss_list) == 5:
		# 	if abs(sum(loss_list)/len(loss_list) - train_loss) < 0.005:
		# 		break
		# 	loss_list[epoch_i%len(loss_list)] = train_loss
		# else:
		# 	loss_list.append(train_loss)

	f.close()


if __name__ == '__main__':
	start1 = time.time()
	print(torch.__version__)
	print(torch.version.cuda)
	print(torch.backends.cudnn.version())
	for i in range(10):
		with open('xxxxx_pro_flows_%d_noip_fold.pkl'%i, 'rb') as f:
			flow_dict = pickle.load(f)
		print('====', i, ' fold validation ====')
		main(i, flow_dict)
	end1 = time.time()
	print('Using time is : ',str(end1-start1))