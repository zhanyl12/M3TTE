# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-30 15:58:51
# @Last Modified by:   xiegr
# @Last Modified time: 2020-09-18 14:22:48
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

n = 3
packet_number = 10


torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True

class SelfAttention(nn.Module):
	"""docstring for SelfAttention"""
	def __init__(self, d_dim=256, dropout=0.1):
		super(SelfAttention, self).__init__()
		# for query, key, value, output
		self.dim = d_dim
		self.linears = nn.ModuleList([nn.Linear(d_dim, d_dim) for _ in range(4)])
		self.dropout = nn.Dropout(p=dropout)

	def attention(self, query, key, value):
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
		scores = F.softmax(scores, dim=-1)
		return scores

	def forward(self, query, key, value):
		# 1) query, key, value
		query, key, value = \
		[l(x) for l, x in zip(self.linears, (query, key, value))]

		# 2) Apply attention
		scores = self.attention(query, key, value)
		x = torch.matmul(scores, value)

		# 3) apply the final linear
		x = self.linears[-1](x.contiguous())
		# sum keepdim=False
		return self.dropout(x), torch.mean(scores, dim=-2)

class OneDimCNN(nn.Module):
	"""docstring for OneDimCNN"""
	# https://blog.csdn.net/sunny_xsc1994/article/details/82969867
	def __init__(self, max_byte_len, d_dim=256, \
		kernel_size = [3, 4], filters=256, dropout=0.1):
		super(OneDimCNN, self).__init__()
		self.kernel_size = kernel_size
		self.convs = nn.ModuleList([
						nn.Sequential(nn.Conv1d(in_channels=d_dim, 
												out_channels=filters, 
												kernel_size=h),
						#nn.BatchNorm1d(num_features=config.feature_size), 
						nn.ReLU(),
						# MaxPool1d: 
						# stride – the stride of the window. Default value is kernel_size
						nn.MaxPool1d(kernel_size=max_byte_len-h+1))
						for h in self.kernel_size
						]
						)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		out = [conv(x.transpose(-2,-1)) for conv in self.convs]
		out = torch.cat(out, dim=1)
		out = out.view(-1, out.size(1))
		return self.dropout(out)


class SAM(nn.Module):
	"""docstring for SAM"""
	# total header bytes 24
	def __init__(self, num_class, max_byte_len, kernel_size = [3, 4], \
		d_dim=256, dropout=0.1, filters=256):
		super(SAM, self).__init__()
		self.posembedding = nn.Embedding(num_embeddings=max_byte_len*10, 
								embedding_dim=d_dim)
		self.byteembedding = nn.Embedding(num_embeddings=300, 
								embedding_dim=d_dim)
		self.attention = SelfAttention(d_dim, dropout)
		self.cnn = OneDimCNN(max_byte_len*10, d_dim, kernel_size, filters, dropout)
		self.fc = nn.Linear(in_features=256*len(kernel_size),
                            out_features=num_class)

	def forward(self,direction,time1, x, y):
		out = self.byteembedding(x) + self.posembedding(y)
		out, score = self.attention(out, out, out)
		out = self.cnn(out)
		out = self.fc(out)
		if not self.training:
			return F.softmax(out, dim=-1).max(1)[1]#, score
		return out

class Multitask_multimodal(nn.Module):
    def __init__(self, num_class1, num_class2, max_byte_len, kernel_size = [3, 4], \
        d_dim=256, dropout=0.1, filters=512):
        super(Multitask_multimodal, self).__init__()
        
        self.posembedding1 = nn.Embedding(num_embeddings=max_byte_len*packet_number, 
                                embedding_dim=d_dim)
        self.byteembedding1 = nn.Embedding(num_embeddings=300, 
                                embedding_dim=d_dim)
        self.attention1 = SelfAttention(d_dim, dropout)
        self.cnn1 = OneDimCNN(max_byte_len*packet_number, d_dim, kernel_size, filters, dropout)
        
        self.fc1 = nn.Linear(in_features=512*2,
                            out_features=256)
        self.fc2 = nn.Linear(in_features=256,
                            out_features=num_class1)
        #self.fc2 = nn.Linear(in_features=330,
        #                    out_features=num_class1)
        self.fc3 = nn.Linear(in_features=256,
                            out_features=num_class2)
        
        self.byteembedding2 = nn.Embedding(num_embeddings=6100, 
                                embedding_dim=32)
        self.attention2 = SelfAttention(32, dropout)
        self.cnn2 = OneDimCNN(packet_number, 32, kernel_size, 32, dropout)


        self.fc4 = nn.Linear(in_features=32*2,
                            out_features=num_class1)
        self.fc5 = nn.Linear(in_features=32*2,
                            out_features=num_class2)

        
        self.fc6 = nn.Linear(in_features=packet_number,
                            out_features=num_class1)
        self.fc7 = nn.Linear(in_features=packet_number,
                            out_features=num_class2)
        
        self.posembedding1 = nn.Embedding(num_embeddings=max_byte_len*packet_number, 
                                embedding_dim=d_dim)
        self.byteembedding1 = nn.Embedding(num_embeddings=300, 
                                embedding_dim=d_dim)
        self.attention1 = SelfAttention(d_dim, dropout)
        self.cnn1 = OneDimCNN(max_byte_len*packet_number, d_dim, kernel_size, filters, dropout)
        
        self.fc1 = nn.Linear(in_features=512*2,
                            out_features=256)
        
        self.byteembedding2 = nn.Embedding(num_embeddings=6100, 
                                embedding_dim=32)
        self.attention2 = SelfAttention(32, dropout)
        self.cnn2 = OneDimCNN(packet_number, 32, kernel_size, 32, dropout)


        self.fc2 = nn.Linear(in_features=32*2,
                            out_features=16)
        
        self.fc3 = nn.Linear(in_features=packet_number,
                            out_features=16)
        self.fc4 = nn.Linear(in_features=288,
                            out_features=64)
        self.fc5 = nn.Linear(in_features=288,
                            out_features=64)
        self.fc6 = nn.Linear(in_features=288,
                            out_features=64)
        self.fc7 = nn.Linear(in_features=64,
                            out_features=num_class1)
        self.fc8 = nn.Linear(in_features=64,
                            out_features=num_class2)
        


    def forward(self, direction_length, time, x, y):
        
        #1
        
        out = self.byteembedding1(x) + self.posembedding1(y)
        #print(out.shape)
        out, score = self.attention1(out, out, out)
        out = self.cnn1(out)
        out = self.fc1(out)
        
        out11 = self.fc2(out)
        out12 = self.fc3(out)
        
        out = self.byteembedding2(direction_length)
        #print(out.shape)
        out, score = self.attention2(out, out, out)
        out = self.cnn2(out)
        #print(out.shape)
        out21 = self.fc4(out)
        out22 = self.fc5(out)
        #print(x.shape,time.shape)
        
        out31 = self.fc6(time.to(torch.float32))
        out32 = self.fc7(time.to(torch.float32))
        out1 = out11 + out21 + out31
        out2 = out12 + out22 + out32

        if not self.training:
            return F.softmax(out1, dim=-1).max(1)[1], F.softmax(out2, dim=-1).max(1)[1]
        return out1,out2
        
        #2
        out = self.byteembedding1(x) + self.posembedding1(y)
        #print(out.shape)
        out, score = self.attention1(out, out, out)
        out = self.cnn1(out)
        t1 = self.fc1(out)
        
        out = self.byteembedding2(direction_length)
        #print(out.shape)
        out, score = self.attention2(out, out, out)
        out = self.cnn2(out)
        t2 = self.fc2(out)
        
        t3 = self.fc3(time.to(torch.float32))

        t = torch.cat([t1.unsqueeze(dim=2),t2.unsqueeze(dim=2),t3.unsqueeze(dim=2)],dim=1).squeeze()
        #print(t.shape)
        x0 = self.fc4(t)
        x1 = self.fc5(t)
        x2 = self.fc6(t)
        out1 = self.fc7(x0+x1)
        out2 = self.fc8(x0+x2)

        if not self.training:
            return F.softmax(out1, dim=-1).max(1)[1], F.softmax(out2, dim=-1).max(1)[1]
        return out1,out2




class Expert(nn.Module):
    def __init__(self,input_dim,output_dim): #input_dim代表输入维度，output_dim代表输出维度
        super(Expert, self).__init__()
        
        p=0.1
        expert_hidden_layers = [64,32]
        self.expert_layer = nn.Sequential(
                            nn.Linear(input_dim, expert_hidden_layers[0]),
                            nn.ReLU(),
                            nn.Dropout(p),
                            nn.Linear(expert_hidden_layers[0], expert_hidden_layers[1]),
                            nn.ReLU(),
                            nn.Dropout(p),
                            nn.Linear(expert_hidden_layers[1],output_dim),
                            nn.ReLU(),
                            nn.Dropout(p)
                            )  

    def forward(self, x):
        out = self.expert_layer(x)
        return out

class Expert_Gate(nn.Module):
    def __init__(self,feature_dim,expert_dim,n_expert,n_task,use_gate=True): #feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)  use_gate：是否使用门控，如果不使用则各个专家取平均
        super(Expert_Gate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        
        '''专家网络'''
        for i in range(n_expert):
            setattr(self, "expert_layer"+str(i+1), Expert(feature_dim,expert_dim)) 
        self.expert_layers = [getattr(self,"expert_layer"+str(i+1)) for i in range(n_expert)]#为每个expert创建一个DNN
        
        '''门控网络'''
        for i in range(n_task):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(nn.Linear(feature_dim, n_expert),
                                                               nn.Softmax(dim=1))) 
        self.gate_layers = [getattr(self,"gate_layer"+str(i+1)) for i in range(n_task)]#为每个gate创建一个lr+softmax
        
    def forward(self, x):
        x = x.to(torch.float32)
        if self.use_gate:
            # 构建多个专家网络
            E_net = [expert(x) for expert in self.expert_layers]
            E_net = torch.cat(([e[:,np.newaxis,:] for e in E_net]),dim = 1) # 维度 (bs,n_expert,expert_dim)

            # 构建多个门网络
            gate_net = [gate(x) for gate in self.gate_layers]     # 维度 n_task个(bs,n_expert)

            # towers计算：对应的门网络乘上所有的专家网络
            towers = []
            for i in range(self.n_task):
                g = gate_net[i].unsqueeze(2)  # 维度(bs,n_expert,1)
                tower = torch.matmul(E_net.transpose(1,2),g)# 维度 (bs,expert_dim,1)
                towers.append(tower.transpose(1,2).squeeze(1))           # 维度(bs,expert_dim)
        else:
            E_net = [expert(x) for expert in self.expert_layers]
            towers = sum(E_net)/len(E_net)
        return towers


class MMoE(nn.Module):
    #feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)
    #def __init__(self,feature_dim,expert_dim,n_expert,n_task,use_gate=True): 
    def __init__(self,num_class1,num_class2,feature_dim,expert_dim,n_expert,n_task,use_gate=True):
        super(MMoE, self).__init__()
        self.use_gate = use_gate
        self.Expert_Gate = Expert_Gate(feature_dim=feature_dim,expert_dim=expert_dim,n_expert=n_expert,n_task=n_task,use_gate=use_gate)
        
        '''Tower1'''
        p1 = 0 
        hidden_layer1 = [64,32] #[64,32] 
        self.tower1 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer1[0]),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(hidden_layer1[0], hidden_layer1[1]),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(hidden_layer1[1], num_class1))
        '''Tower2'''
        p2 = 0
        hidden_layer2 = [64,32]
        self.tower2 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer2[0]),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_layer2[0], hidden_layer2[1]),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_layer2[1], num_class2))
        
    def forward(self, direction_length, time, x, y):
        #print(x.shape,direction_length.shape,time.shape)
        #print(x.unsqueeze(dim=2).shape,direction_length.unsqueeze(dim=2).shape,time.unsqueeze(dim=2).shape)
        x = torch.cat([x.unsqueeze(dim=2),direction_length.unsqueeze(dim=2),time.unsqueeze(dim=2)],dim=1)
        #print(x.shape)
        x=x.squeeze()
        #print(x.shape)
        towers = self.Expert_Gate(x)
        if self.use_gate:            
            out1 = self.tower1(towers[0])
            out2 = self.tower2(towers[1]) 
        else:
            out1 = self.tower1(towers)
            out2 = self.tower2(towers)
        if not self.training:
            return F.softmax(out1, dim=-1).max(1)[1], F.softmax(out2, dim=-1).max(1)[1]
        return out1,out2


class Expert1(nn.Module):#for x y
    def __init__(self,max_byte_len, expert_dim,kernel_size = [3, 4], \
        d_dim=256, dropout=0.1, filters=512): #input_dim代表输入维度，output_dim代表输出维度
        super(Expert1, self).__init__()
        self.posembedding1 = nn.Embedding(num_embeddings=max_byte_len*10, 
                                embedding_dim=d_dim)
        self.byteembedding1 = nn.Embedding(num_embeddings=300, 
                                embedding_dim=d_dim)
        self.attention1 = SelfAttention(d_dim, dropout)
        self.cnn1 = OneDimCNN(max_byte_len*10, d_dim, kernel_size, filters, dropout)

        self.fc1 = nn.Linear(in_features=512*2,
                            out_features=256)
        self.fc2 = nn.Linear(in_features=256,
                            out_features=expert_dim)

    def forward(self, x, y):
        #print(x)
        #print(y)
        a,b = x.shape
        #return torch.zeros(a,64).cuda()
        out = self.byteembedding1(x) + self.posembedding1(y)
        #print(out.shape)
        out, score = self.attention1(out, out, out)
        out = self.cnn1(out)

        out = self.fc1(out)
        out = self.fc2(out)
        return out

class Expert2(nn.Module):# for direction_length and time
    def __init__(self,max_byte_len, packet_number,expert_dim,kernel_size = [3, 4], \
        #d_dim=256, dropout=0.1, filters=512): #input_dim代表输入维度，output_dim代表输出维度
        d_dim=32, dropout=0.1, filters=32):
        super(Expert2, self).__init__()
        
        
        #attention
        self.byteembedding1 = nn.Embedding(num_embeddings=6100, 
                                embedding_dim=d_dim)
        #self.poseembedding1 = nn.Embedding(num_embeddings=500, 
        #                        embedding_dim=d_dim)
        self.attention1 = SelfAttention(d_dim, dropout)
        self.cnn1 = OneDimCNN(packet_number, d_dim, kernel_size, filters, dropout)


        self.fc1 = nn.Linear(in_features=32*2,
                            out_features=expert_dim)
        self.fc2 = nn.Linear(in_features=256,
                            out_features=expert_dim)
        
        
        '''
        #LSTM method
        self.hidden_size = 64
        self.lstm1 = nn.LSTM(packet_number, self.hidden_size, 2, bidirectional=True, batch_first=True)
        self.w1 = nn.Parameter(torch.zeros(self.hidden_size * 2))
        self.fc1  = nn.Linear(self.hidden_size * 2, 64)
        self.lstm2 = nn.LSTM(packet_number, self.hidden_size, 2, bidirectional=True, batch_first=True)
        self.w2 = nn.Parameter(torch.zeros(self.hidden_size * 2))
        self.fc2  = nn.Linear(self.hidden_size * 2, 64)
        self.fc3  = nn.Linear(128, expert_dim)
        '''


    def forward(self, direction_length,time):
        #attention
        #a,b = direction_length.shape
        #return torch.zeros(a,64).cuda()
        #print(direction_length.shape)
        #print(time.shape)
        #print(direction_length)
        #print(direction_length)
        #print(direction_length.shape)
        
        out = self.byteembedding1(direction_length)
        #print(out.shape)
        #out = out + time*100
        out, score = self.attention1(out, out, out)
        out = self.cnn1(out)

        out = self.fc1(out)
        #out = self.fc2(out)
        return out
        
        '''
        #LSTM
        #print(direction_length.to(torch.float32).shape)
        #print(direction_length.to(torch.float32).unsqueeze(dim=1).shape)
        H1, _ = self.lstm1(direction_length.to(torch.float32).unsqueeze(dim=1))
        #print(torch.matmul(H1, self.w1).shape)
        alpha1 = F.softmax(torch.matmul(H1, self.w1), dim=1).unsqueeze(-1)  
        out1 = H1 * alpha1  
        out1 = torch.sum(out1, 1)
        out1 = F.relu(out1)
        out1 = self.fc1(out1)
        
        H2, _ = self.lstm2(time.to(torch.float32).unsqueeze(dim=1))  
        alpha2 = F.softmax(torch.matmul(H2, self.w2), dim=1).unsqueeze(-1)  
        out2 = H2 * alpha2
        out2 = torch.sum(out2, 1)
        out2 = F.relu(out2)
        out2 = self.fc2(out2)
        #print(out1.shape)
        #print(out2.shape)
        out = torch.cat([out1,out2],dim=1)
        #print(out.shape)
        out = self.fc3(out)
        return out
        '''

class Expert3(nn.Module):# for direction_length and time
    def __init__(self,packet_number,expert_dim,kernel_size = [3, 4], \
        #d_dim=256, dropout=0.1, filters=512): #input_dim代表输入维度，output_dim代表输出维度
        d_dim=32, dropout=0.1, filters=32):
        super(Expert3, self).__init__()
        self.fc1 = nn.Linear(in_features=packet_number,
                            out_features=expert_dim)

    def forward(self, direction_length,time):
        out = self.fc1(time.to(torch.float32))
        #out = self.fc2(out)
        return out


class Multimodal_Expert_Gate(nn.Module):
    def __init__(self,max_byte_len,expert_dim,n_expert,n_task,use_gate=True): #feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)  use_gate：是否使用门控，如果不使用则各个专家取平均
        super(Multimodal_Expert_Gate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate
        
        '''专家网络'''
        setattr(self, "expert_layer1", Expert1(max_byte_len,expert_dim))
        setattr(self, "expert_layer2", Expert2(max_byte_len,packet_number,expert_dim))
        setattr(self, "expert_layer3", Expert3(packet_number,expert_dim))
        self.expert_layers = [getattr(self,"expert_layer1"),getattr(self,"expert_layer2"),getattr(self,"expert_layer3")]#为每个expert创建一个DNN
        
        '''
        门控网络
        for i in range(n_task):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(nn.Linear(256, n_expert),
                                                               nn.Softmax(dim=1))) 
        self.gate_layers = [getattr(self,"gate_layer"+str(i+1)) for i in range(n_task)]#为每个gate创建一个lr+softmax
        '''
        '''
        self.gate_layer1=nn.Sequential(nn.Linear(500+packet_number*2, n_expert),nn.Softmax(dim=1))
        self.gate_layer2=nn.Sequential(nn.Linear(500+packet_number*2, n_expert),nn.Softmax(dim=1))
        self.gate_layer3=nn.Sequential(nn.Linear(500+packet_number*2, n_expert),nn.Softmax(dim=1))
        '''

        self.gate_layer1=nn.Sequential(nn.Linear(500, n_expert),nn.Softmax(dim=1))
        self.gate_layer2=nn.Sequential(nn.Linear(packet_number, n_expert),nn.Softmax(dim=1))
        
    def forward(self, direction_length, time, x, y):
        if self.use_gate:
            # 构建多个专家网络
            E_net = [self.expert_layers[0](x,y),self.expert_layers[1](direction_length,time),self.expert_layers[2](direction_length,time)]
            E_net = torch.cat(([e[:,np.newaxis,:] for e in E_net]),dim = 1) # 维度 (bs,n_expert,expert_dim)

            # 构建多个门网络
            #gate_net = [gate(x.float()) for gate in self.gate_layers]     # 维度 n_task个(bs,n_expert)
            gate_net = [self.gate_layer1(x.float()+y.float()),self.gate_layer2(time.float())]
            #now = torch.cat([x.unsqueeze(dim=2),direction_length.unsqueeze(dim=2),time.unsqueeze(dim=2)],dim=1).squeeze()
            #gate_net = [self.gate_layer1(now.float()),self.gate_layer2(now.float()),self.gate_layer3(now.float())]
            # towers计算：对应的门网络乘上所有的专家网络
            towers = []
            for i in range(self.n_task):
                g = gate_net[i].unsqueeze(2)  # 维度(bs,n_expert,1)
                tower = torch.matmul(E_net.transpose(1,2),g)# 维度 (bs,expert_dim,1)
                towers.append(tower.transpose(1,2).squeeze(1))           # 维度(bs,expert_dim)
        else:
            E_net = [expert(x) for expert in self.expert_layers]
            towers = sum(E_net)/len(E_net)
        return towers


class Multimodal_MMoE(nn.Module):
    #feature_dim:输入数据的维数  expert_dim:每个神经元输出的维数  n_expert:专家数量  n_task:任务数(gate数)
    #def __init__(self,feature_dim,expert_dim,n_expert,n_task,use_gate=True): 
    def __init__(self,num_class1,num_class2,max_byte_len,expert_dim,n_expert,n_task,use_gate=True):
        super(Multimodal_MMoE, self).__init__()
        self.use_gate = use_gate
        self.Multimodal_Expert_Gate = Multimodal_Expert_Gate(max_byte_len=max_byte_len,expert_dim=expert_dim,n_expert=n_expert,n_task=n_task,use_gate=use_gate)
        '''Tower1'''
        p1 = 0 
        hidden_layer1 = [64,32] #[64,32] 
        self.tower1 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer1[0]),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(hidden_layer1[0], hidden_layer1[1]),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(hidden_layer1[1], num_class1))
        '''Tower2'''
        p2 = 0
        hidden_layer2 = [64,32]
        self.tower2 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer2[0]),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_layer2[0], hidden_layer2[1]),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_layer2[1], num_class2))
        
    def forward(self, direction_length, time, x, y):
        #print(direction_length.shape)
        #a,b = direction_length.shape
        #temp = torch.zeros(a,b)
        towers = self.Multimodal_Expert_Gate(direction_length, time, x, y)
        #towers = self.Multimodal_Expert_Gate(temp, temp, x, y)
        if self.use_gate:            
            out1 = self.tower1(towers[0])
            out2 = self.tower2(towers[1]) 
        else:
            out1 = self.tower1(towers)
            out2 = self.tower2(towers)
        if not self.training:
            return F.softmax(out1, dim=-1).max(1)[1], F.softmax(out2, dim=-1).max(1)[1]
        return out1,out2


class FSSAM(nn.Module):
	"""docstring for SAM"""
	# total header bytes 24
	def __init__(self, num_class, max_byte_len, kernel_size = [3, 4], \
		d_dim=256, dropout=0.1, filters=512):
		super(FSSAM, self).__init__()
		self.posembedding = nn.Embedding(num_embeddings=max_byte_len*8, 
								embedding_dim=d_dim)
		self.byteembedding = nn.Embedding(num_embeddings=300, 
								embedding_dim=d_dim)
		self.attention = SelfAttention(d_dim, dropout)
		self.cnn = OneDimCNN(max_byte_len*8, d_dim, kernel_size, filters, dropout)
		self.fc = nn.Linear(in_features=512*len(kernel_size),
                            out_features=num_class)

	def forward(self, x, y):
		out = self.byteembedding(x) + self.posembedding(y)
		out, score = self.attention(out, out, out)
		out = self.cnn(out)
		out = self.fc(out)
		if not self.training:
			return F.softmax(out, dim=-1).max(1)[1], score
		return out


class OneCNN(nn.Module):
    def __init__(self,label_num):
        super(OneCNN,self).__init__()
        self.layer_1 = nn.Sequential(
            # 输入784*1
            nn.Conv2d(1,32,(1,25),1,padding='same'),
            nn.ReLU(),
            # 输出262*32
            nn.MaxPool2d((1, 3), 3, padding=0),
        )
        self.layer_2 = nn.Sequential(
            # 输入261*32
            nn.Conv2d(32,64,(1,25),1,padding='same'),
            nn.ReLU(),
            # 输入261*64
            nn.MaxPool2d((1, 3), 3, padding=0)
        )
        self.fc1=nn.Sequential(
            # 输入88*64
            nn.Flatten(),
            nn.Linear(3520,1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024,label_num),
            nn.Dropout(p=0.3)
        )
    def forward(self,direction,time1,x,y):
        #print("x.shape:",x.shape)
        x = x.to(torch.float32)
        x=self.layer_1(x)
        #print("x.shape:",x.shape)
        x=self.layer_2(x)
        #print("x.shape:",x.shape)
        x=self.fc1(x)
        #print("x.shape:",x.shape)
        if not self.training:
            return F.softmax(x, dim=-1).max(1)[1]
        return x

class DatanetMLP(nn.Module):
    def __init__(self,label_num):
        super(DatanetMLP, self).__init__()
    
        self.fc1 = nn.Linear(in_features=500, out_features=128) 
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(in_features=32, out_features= label_num)
        self.dropout = nn.Dropout(0.3)
        self.f1 = nn.Flatten()
    def forward(self,direction,time,x,y):
        #print("x.shape:",x.shape)
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        x = self.f1(x)
        #print("x.shape:",x.shape)
        if not self.training:
        	return F.softmax(x, dim=-1).max(1)[1]
        return x


class DatanetCNN(nn.Module):
    def __init__(self,label_num):
        super(DatanetCNN,self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              # 灰度图片的高度为1，input height
                out_channels=8,            # 16个卷积，之后高度为从1变成16，长宽不变，n_filters
                kernel_size=5,              # 5*5宽度的卷积，filter size
                stride=1,                   # 步幅为1，filter movement/step
                padding=2,                  # 周围填充2圈0，if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # 激活时，图片长宽高不变，activation
            nn.MaxPool2d(kernel_size=2),    # 4合1的池化，之后图片的高度不变，长宽减半，choose max value in 2x2 
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),    
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.fc1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(192,1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024,label_num),
            nn.Dropout(p=0.3)
        )
    def forward(self,direction,time1,x,y):
        #print("x.shape:",x.shape)
        x = x.to(torch.float32)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        #print("x.shape:",x.shape)
        x = self.fc1(x)
        if not self.training:
        	return F.softmax(x, dim=-1).max(1)[1]
        return x


class BiLSTM(nn.Module):
    def __init__(self,label_num):
        super(BiLSTM, self).__init__()
        self.hidden_size = 128
        self.lstm = nn.LSTM(500, self.hidden_size, 2, bidirectional=True, batch_first=True)
        self.w = nn.Parameter(torch.zeros(self.hidden_size * 2))
        self.fc1  = nn.Linear(self.hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, label_num)
    def forward(self,direction,time1,x,y):
        #print("x.shape:",x.shape)
        x = x.to(torch.float32)
        H, _ = self.lstm(x)  
        #print('H.size is : ',H.shape)
        alpha = F.softmax(torch.matmul(H, self.w), dim=1).unsqueeze(-1)  
        out = H * alpha  
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)  
        if not self.training:
        	return F.softmax(out, dim=-1).max(1)[1]
        return out


class DeepPacket(nn.Module):
    def __init__(self,label_num):
        super(DeepPacket, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=200,kernel_size=4,stride=3)
        self.conv2 = nn.Conv1d(in_channels=200,out_channels=200,kernel_size=5,stride=1)
        
        #self.fc1 = nn.Linear(in_features=200*128, out_features=200) # ((28-5+1)/2 -5 +1)/2 = 4
        self.fc1 = nn.Linear(in_features=200*81, out_features=200)
        self.dropout = nn.Dropout(0.05)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.dropout = nn.Dropout(0.05)
        self.fc3 = nn.Linear(in_features=100, out_features=50)
        self.dropout = nn.Dropout(0.05)
        self.out = nn.Linear(in_features=50, out_features= label_num)
    def forward(self,direction,time,x,y):
        x = x.to(torch.float32)
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.max_pool1d(out, kernel_size=2)
        #print('out shape is:',out.shape)
        out = out.reshape(-1, 200*81) 
        
        out = self.fc1(out)
        out = self.dropout(out)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = self.dropout(out)
        out = F.relu(out)

        out = self.out(out)
        if not self.training:
        	return F.softmax(out, dim=-1).max(1)[1]
        return out


class TSCRNN(nn.Module):
    def __init__(self,label_num):
        super(TSCRNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels= 1, out_channels= 64 , kernel_size=3, stride=1, padding=1)
        #in channels 输入矩阵的行数
        self.bn1 = nn.BatchNorm1d(64,affine = True)
        self.conv2 = nn.Conv1d(in_channels= 64, out_channels= 64 , kernel_size=3, stride=1,  padding=1)
        self.bn2 = nn.BatchNorm1d(64,affine = True)
        self.lstm = nn.LSTM(125, 256, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(in_features=512, out_features=label_num)
    def forward(self,direction,time1,x,y):
        #print("x.shape:",x.shape)
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x,_ = self.lstm(x)
        x = self.dropout(x)
        
        x = self.out(x[:, -1, :]) 
        #print("x.shape:",x.shape)
        if not self.training:
        	return F.softmax(x, dim=-1).max(1)[1]
        return x

class WF(nn.Module):
    def __init__(self, nb_classes):
        super(WF, self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv1d(1, 32, 8, 1, 0),
            nn.BatchNorm1d(32), 
            nn.ReLU(),                     
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1), 
        )

        self.conv2 = nn.Sequential(       
            nn.Conv1d(32, 64, 8, 1, 0),
            nn.BatchNorm1d(64), 
            nn.ReLU(),                     
            nn.MaxPool1d(8, 4, 0),
            nn.Dropout(0.1), 
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1856, 256),#number to change
            nn.ReLU(),
            nn.Linear(256, nb_classes)
        )   

    def forward(self, direction_length, time, x, y):
        x = torch.cat([direction_length.unsqueeze(dim=2),time.unsqueeze(dim=2),y.unsqueeze(dim=2)],dim=1)
        #print(x.shape)
        x=x.squeeze()
        #print(x.shape)
        x=x.unsqueeze(dim=1)
        #print(x.shape)
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)     
        output = self.out(x)
        if not self.training:
            return F.softmax(output, dim=-1).max(1)[1]
        return output


if __name__ == '__main__':
	x = np.random.randint(0, 255, (10, 20))
	y = np.random.randint(0, 20, (10, 20))
	sam = SAM(num_class=5, max_byte_len=20)
	out = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
	print(out[0])

	sam.eval()
	out, score = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
	print(out[0], score[0])
