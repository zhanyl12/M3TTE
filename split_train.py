import pickle
import dpkt
import random
import numpy as np
from tqdm import tqdm, trange


cdn_kind = ['origin', 'ali', 'tencent', 'baidu', 'cloudflare', 'cloudfront','qiniu', 'fastly', 'self']
websites_kind = ['Blog','Picture','Video','BBS','Social']

ip_features = {'hl':1,'tos':1,'len':2,'df':1,'mf':1,'ttl':1,'p':1}
tcp_features = {'off':1,'flags':1,'win':2}
udp_features = {'ulen':2}
max_byte_len = 50
n = 10

def mask(p):
	p.src = b'\x00\x00\x00\x00'
	p.dst = b'\x00\x00\x00\x00'
	p.sum = 0
	p.id = 0
	p.offset = 0

	if isinstance(p.data, dpkt.tcp.TCP):
		p.data.sport = 0
		p.data.dport = 0
		p.data.seq = 0
		p.data.ack = 0
		p.data.sum = 0

	elif isinstance(p.data, dpkt.udp.UDP):
		p.data.sport = 0
		p.data.dport = 0
		p.data.sum = 0

	return p

def pkt2feature(data, k):
	max_number = 0
	min_number = 1000
	flow_dict = {'train':[], 'test':[] }
	#print(k)
	# train->protocol->flowid->[pkts]
	flow_dict['train'] = []
	flow_dict['test'] = []
	pkts_after = []
	feature_vector = []
	all_number = int(len(data))
	#print('All number is : ',all_number)
	#below do the flow sampling masking and discard the short flows
	for each in data:
		if len(each[1])<=15:
			continue  #discard the short flows
		now_feature = []
		feature_1 = []
		feature_2 = []
		now_feature.append([int(each[0][0]),int(each[0][1])])
		packet_number = int(each[3])
		
		
		#method 1 first n
		for i in range(n):
			feature_1.append(each[1][i])
			pkt = mask(each[2][i])
			raw_byte = pkt.pack()
			feature_2.append(raw_byte)
		now_feature.append(feature_1)
		now_feature.append(feature_2)
		now_feature.append(packet_number)
		
		#method 2 random n
		'''
		counting_number = 0
		last_number = -1
		while counting_number<n:
			choose_number = random.randint(0, packet_number-1)
			if choose_number != last_number:
				last_number = choose_number
				feature_1.append(each[1][choose_number])
				pkt = mask(each[2][choose_number])
				raw_byte = pkt.pack()
				feature_2.append(raw_byte)
				counting_number = counting_number + 1
			else:
				continue
		now_feature.append(feature_1)
		now_feature.append(feature_2)
		now_feature.append(packet_number)
		'''
		#method 3 random lianxu n
		'''
		choose_number = random.randint(0,packet_number-1-n)
		for i in range(n):
			feature_1.append(each[1][choose_number+i])
			pkt = mask(each[2][choose_number+i])
			raw_byte = pkt.pack()
			feature_2.append(raw_byte)
		now_feature.append(feature_1)
		now_feature.append(feature_2)
		now_feature.append(packet_number)
		'''
		#method 4 mid n
		'''
		for i in range(n):
			feature_1.append(each[1][int(packet_number/2)+i])
			pkt = mask(each[2][int(packet_number/2)+i])
			raw_byte = pkt.pack()
			feature_2.append(raw_byte)
		now_feature.append(feature_1)
		now_feature.append(feature_2)
		now_feature.append(packet_number)
		'''
		
		feature_vector.append(now_feature)
	#below split the train and the test dataset
	
	for idx in range(len(feature_vector)):
		label1 = int(feature_vector[idx][0][0])
		label2 = int(feature_vector[idx][0][1])
		#print(label1,label2)
		#print(feature_vector[idx][3])
		direction_length_list = []
		time_list = []
		byte = []
		pos = []
		for i in range(n):
			#print(feature_vector[idx][1])
			direction_length_list.append(int(feature_vector[idx][1][i][0])*int(feature_vector[idx][1][i][1])+1800)
			temp = int(feature_vector[idx][1][i][0])*int(feature_vector[idx][1][i][1])+1800
			if temp < min_number:
				min_number = temp
			if temp > max_number:
				max_number = temp
			time_list.append(float(feature_vector[idx][1][i][2]))
			for x in range(min(len(feature_vector[idx][2][i]),max_byte_len)):
				byte.append(int(feature_vector[idx][2][i][x]))
				pos.append(i*max_byte_len + x)
			byte.extend([0]*((i+1)*max_byte_len-len(byte)))
			pos.extend([0]*((i+1)*max_byte_len-len(pos)))
		#print (len(byte),len(pos))
		#print('=====================================')
		#print(label1,label2)
		#print(label1, label2, len(direction_length_list),len(time_list),len(byte),len(pos))
		#print(direction_length_list)
		#print(time_list)
		#print(pos)
		if idx in range(k*int(len(feature_vector)*0.1), (k+1)*int(len(feature_vector)*0.1)):
			flow_dict['test'].append((label1, label2, direction_length_list, time_list, byte, pos))
		else:
			flow_dict['train'].append((label1, label2, direction_length_list, time_list, byte, pos))
	print(max_number,min_number)
	return flow_dict


def load_epoch_data(flow_dict, train='train'):
	flow_list = flow_dict[train]
	label1_list, label2_list, direction_length_list, time_list, x, y = [], [], [], [], [], []

	for each in flow_list:
		#print(each)
		#print(each[0],each[1])
		'''
		for label1, label2, direction_length, time, byte, pos in each:
			label1_list.append(label1)
			label2_list.append(label2)
			direction_length_list.append(direction_length)
			time_list.append(time)
			x.append(byte)
			y.append(pos)
		'''
		label1_list.append(each[0])
		label2_list.append(each[1])
		direction_length_list.append(each[2])
		time_list.append(each[3])
		x.append(each[4])
		y.append(each[5])

	return np.array(label1_list)[:, np.newaxis], np.array(label2_list)[:, np.newaxis], np.array(direction_length_list), np.array(time_list), np.array(x), np.array(y)


if __name__ == '__main__':
	hehe=[10]
	with open('list_flows.pkl','rb') as f:
		data = pickle.load(f)
	print('Finish loading...')
	for each in hehe:
		n = int(each)
		print ('n is : ',n)
		for i in trange(10, mininterval=2, \
			desc='  - (Building fold dataset)   ', leave=False):
			print(n,i)
			flow_dict = pkt2feature(data, i)
			with open('xxxxx'+'_pro_flows_%d_noip_fold.pkl'%i, 'wb') as f:
				pickle.dump(flow_dict, f)