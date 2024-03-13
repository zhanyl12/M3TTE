import numpy as np
import dpkt
import random
import pickle
import os
from scapy.all import * 

cdn_kind = ['origin', 'ali', 'tencent', 'baidu', 'cloudflare', 'cloudfront','qiniu', 'fastly', 'self']
website_kind = ['Blog','Picture','Video','BBS','Social']
data_path = 'C://research//all_data_after'


def get_flows():
	filenames=os.listdir(data_path)
	flows = {}
	result_list = []
	for files in filenames:
		if 'pcapng' in files:
			print(files)
			flows[files.split('.')[0]]=[]
			flows[files.split('.')[0]].append([int(files.strip().split('_')[0]),int(cdn_kind.index(files.strip().split('_')[2]))])
			flows[files.split('.')[0]].append([])
			flows[files.split('.')[0]].append([])
			pcap = dpkt.pcapng.Reader(open(data_path+'//'+files, 'rb'))
			zhanzhan = 0
			basic_time = 0
			basic_address = ''
			for time, buff in pcap:
				#print (time)
				#print (len(buff))
				if zhanzhan >= 800:
					break
				eth = dpkt.ethernet.Ethernet(buff)
				if isinstance(eth.data, dpkt.ip.IP) and (isinstance(eth.data.data, dpkt.udp.UDP)or isinstance(eth.data.data, dpkt.tcp.TCP)):
					# tcp or udp packet
					ip = eth.data
					if zhanzhan == 0:
						basic_time = time
						basic_address = ip.src
					zhanzhan = zhanzhan+1
					#print(ip.src,ip.dst)
					if ip.src == basic_address:
						flows[files.split('.')[0]][1].append([1,len(buff),time-basic_time])
					else:
						flows[files.split('.')[0]][1].append([-1,len(buff),time-basic_time])
					flows[files.split('.')[0]][2].append(ip)
			flows[files.split('.')[0]].append(int(zhanzhan))
			result_list.append(flows[files.split('.')[0]])
	return flows,result_list


if __name__ == '__main__':
	flows,result_list = get_flows()
	web_cal={}
	for each in website_kind:
		web_cal[each]=[0,0]
	cdn_cal={}
	for each in cdn_kind:
		cdn_cal[each]=[0,0]
	for each in flows.keys():
		web_cal[website_kind[flows[each][0][0]]][0]=web_cal[website_kind[flows[each][0][0]]][0] + 1
		web_cal[website_kind[flows[each][0][0]]][1]=web_cal[website_kind[flows[each][0][0]]][1] + flows[each][3]
	for each in flows.keys():
		cdn_cal[cdn_kind[flows[each][0][1]]][0]=cdn_cal[cdn_kind[flows[each][0][1]]][0] + 1
		cdn_cal[cdn_kind[flows[each][0][1]]][1]=cdn_cal[cdn_kind[flows[each][0][1]]][1] + flows[each][3]
	print('Now is Website results : ')
	for each in web_cal.keys():
		print('============================')
		print('Generate flows for %s'%each)
		print('Total flows: ', web_cal[each][0])
		print('Total pkts: ', web_cal[each][1])
	print('========================================================')
	print('Now is CDN results : ')
	for each in cdn_cal.keys():
		print('============================')
		print('Generate flows for %s'%each)
		print('Total flows: ', cdn_cal[each][0])
		print('Total pkts: ', cdn_cal[each][1])
	random.shuffle(result_list)
	with open('pro_flows.pkl', 'wb') as f:
		pickle.dump(flows, f)
	with open('list_flows.pkl', 'wb') as f:
		pickle.dump(result_list, f)
	#print(result_list)
	#with open('pro_flows.pkl','rb') as f:
	#	data = pickle.load(f)
	print('Finish loading...')
	#print(data)
	'''
	a = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	random.shuffle(a)
	print(a)
	random.shuffle(a)
	print(a)
	random.shuffle(a)
	print(a)
	'''