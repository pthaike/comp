#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
# import MySQLdb as mdb
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pdb
import time

# get artist list
def getitem():
	file = 'item2.csv'
	name = pd.read_csv(file, header=None, names = ['id'])
	return name

# # get the product data
# def getall():
# 	item = getitem()
# 	start = pd.to_datetime('20141001')
# 	end = pd.to_datetime('20151227')
# 	# dat = -1*np.ones((len(item),(end-start).days+1))
# 	dat = np.zeros((len(item),(end-start).days+1))
# 	con = mdb.connect('localhost', 'sealyn', 'lyn520', 'bird')
# 	index = 0
# 	for id in item.id:
# 		print id
# 		sql = 'select * from item_buy where item_id = "' + str(id) +'"'
# 		with con:
# 			sqlres = pd.read_sql(sql = sql, con = con)
# 			sqlres.date = pd.to_datetime(sqlres.date,format='%Y%m%d')
# 			for i in range(len(sqlres)):
# 				d = (sqlres.loc[i, 'date']-start).days
# 				dat[index,d] = sqlres.loc[i, 'qty_alipay_njhs']
# 			index += 1
# 	res = pd.DataFrame(dat, index = item.id)
# 	res.to_csv('dat/all.csv')

# """
# sql = 'select date, item_id, pv_ipv from item_feature where item_id = "' + str(id) +'"'
# sql = 'select date, item_id, pv_uv from item_feature where item_id = "' + str(id) +'"'


# """

# def getallbrow():
# 	item = getitem()
# 	start = pd.to_datetime('20141001')
# 	end = pd.to_datetime('20151227')
# 	dat = -1*np.ones((len(item),(end-start).days+1))
# 	# dat = np.zeros((len(item),(end-start).days+1))
# 	con = mdb.connect('localhost', 'sealyn', 'lyn520', 'bird')
# 	index = 0
# 	for id in item.id:
# 		print id
# 		sql = 'select date, item_id, pv_ipv from item_feature where item_id = "' + str(id) +'"'
# 		with con:
# 			sqlres = pd.read_sql(sql = sql, con = con)
# 			sqlres.date = pd.to_datetime(sqlres.date,format='%Y%m%d')
# 			for i in range(len(sqlres)):
# 				d = (sqlres.loc[i, 'date']-start).days
# 				dat[index,d] = sqlres.loc[i, 'qty_alipay_njhs']
# 			index += 1
# 	res = pd.DataFrame(dat, index = item.id)
# 	res.to_csv('dat/na/all_pv_ipv.csv')



# def getstore():
# 	item = getitem()
# 	start = pd.to_datetime('20141001')
# 	end = pd.to_datetime('20151227')
# 	# dat = np.zeros((len(item),(end-start).days+1))
# 	dat = -1*np.ones((len(item),(end-start).days+1))
# 	con = mdb.connect('localhost', 'sealyn', 'lyn520', 'bird')
# 	for store in range(1,6):
# 		index = 0
# 		for id in item.id:
# 			print id, store
# 			sql = 'select * from item_store_buy where item_id = "' + str(id) + '" and store_code = ' + str(store)
# 			with con:
# 				sqlres = pd.read_sql(sql = sql, con = con)
# 				sqlres.date = pd.to_datetime(sqlres.date,format='%Y%m%d')
# 				for i in range(len(sqlres)):
# 					d = (sqlres.loc[i, 'date']-start).days
# 					dat[index,d] = sqlres.loc[i, 'qty_alipay_njhs']
# 				index += 1
# 		res = pd.DataFrame(dat, index = item.id)
# 		res.to_csv('dat/'+str(store)+'.csv')


# def getstorebrow():
# 	item = getitem()
# 	start = pd.to_datetime('20141001')
# 	end = pd.to_datetime('20151227')
# 	start = pd.to_datetime('20141001')
# 	end = pd.to_datetime('20151227')





def getseq(tag):
	file = 'dat/na/'+tag+'.csv'
	dat = pd.read_csv(file, index_col = 0)
	return dat

def getweight():
	file = 'config2.csv'
	dat = pd.read_csv(file, header=None, names = ['id', 'store', 'weight'])
	dict = {}
	for i in dat.index:
		ident = dat.loc[i, 'id']
		store = dat.loc[i, 'store']
		weight = dat.loc[i, 'weight']
		w = weight.split('_')
		pos = float(w[0])
		neg = float(w[1])
		if ident not in dict:
			dict[ident] = {}
		dict[ident][store] = {'pos':pos, 'neg':neg}
	return dict


# """
# seq: sequence
# step: x step
# ahead:

# return dat:train data, x: predict x
# """
# def gemfeature(seq, step, ahead):
# 	# pdb.set_trace()
# 	s = seq[step-1:len(seq)]
# 	m = len(s[s>-1])
# 	dat = np.zeros((m, step+1))
# 	index = 0;
# 	for i in range(m):
# 		y = sum(seq[i+step:i+step+ahead])
# 		if y > -1:
# 			dat[index, 0] = y
# 			dat[index, 1:step+1] = seq[i:i+step]
# 			index += 1
# 	x = seq[m:m+step]
# 	return dat,x

# def gemfeature(seq, step, ahead):
# 	# pdb.set_trace()
# 	s = seq[step-1:len(seq)]
# 	m = len(s[s>-1])
# 	dat = np.zeros((m, step+1))
# 	index = 0;
# 	for i in range(m):
# 		y = sum(seq[i+step:i+step+ahead])
# 		if y > -1:
# 			dat[index, 0] = y
# 			dat[index, 1:step+1] = seq[i:i+step]
# 			index += 1
# 	x = seq[m:m+step]
# 	return dat,x


#old gemfeature
# def gemfeature(seq, step, ahead):
# 	pdb.set_trace()
# 	m = len(seq)-step
# 	dat = np.zeros((m, step+1))
# 	for i in range(m):
# 		dat[i, 0] = sum(seq[i+step:i+step+ahead]) 
# 		dat[i, 1:step+1] = seq[i:i+step]
# 	x = seq[m:m+step]
# 	return dat,x


# def figures():
# 	allseq = getseq('all')
# 	seq1 = getseq('1')
# 	seq2 = getseq('2')
# 	seq3 = getseq('3')
# 	seq4 = getseq('4')
# 	seq5 = getseq('5')
# 	for i in allseq.index:
# 		plt.plot(all,'r')
# 		plt.plot(s1)
# 		plt.plot(s2)
# 		plt.plot(s3)
# 		plt.plot(s4)
# 		plt.plot(s5)
# 		plt.show()


"""
20141111:41
20141212:72
20151111:406
20151212:437
return： 
	 0: return seq
	-1：the data count less than 14
	-2: the latest data are so small
"""
def clean(seq):
	print type(seq)
	# pdb.set_trace()
	seq = seq.drop(seq.index[[41,72,406,437]])
	for i in range(len(seq)):
		if seq[i] != -1:
			break;
	s = seq[i:len(seq)]
	# set missing data to zero
	s[s==-1] = 0
	if len(s)<=21:
		s[s==-1] = 0
		return -1,s
	sl = s[len(s)-21:len(s)]
	if len(sl[sl>0])<7:
		s[s==-1] = 0
		return -2,sl
	return 0,s

def getfromcsv():
	file = 'item_feature2.csv'
	csvdat = pd.read_csv(file, header = None, names = [
		'date',
		'item_id',
		'cate_id',
		'cate_level_id',
		'brand_id',
		'supplier_id',
		'pv_ipv',
		'pv_uv',
		'cart_ipv',
		'cart_uv',
		'collect_uv',
		'num_gmv',
		'amt_gmv',
		'qty_gmv',
		'unum_gmv',
		'amt_alipay',
		'num_alipay',
		'qty_alipay',
		'unum_alipay',
		'ztc_pv_ipv',
		'tbk_pv_ipv',
		'ss_pv_ipv',
		'jhs_pv_ipv',
		'ztc_pv_uv',
		'tbk_pv_uv',
		'ss_pv_uv',
		'jhs_pv_uv',
		'num_alipay_njhs',
		'amt_alipay_njhs',
		'qty_alipay_njhs',
		'unum_alipay_njhs'
		])
	item = getitem()
	start = pd.to_datetime('20141001')
	end = pd.to_datetime('20151227')
	res_qty_alipay_njhs = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_pv_ipv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_pv_uv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_cart_ipv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_cart_uv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_collect_uv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_num_gmv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_qty_gmv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_unum_gmv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)

	res_num_alipay = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_qty_alipay = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_ztc_pv_ipv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_tbk_pv_ipv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_ss_pv_ipv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_jhs_pv_uv = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_num_alipay_njhs = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)
	res_unum_alipay_njhs = pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)


	csvdat.date = pd.to_datetime(csvdat.date,format='%Y%m%d')
	for i in range(len(csvdat.index)):
		print i/float(len(csvdat.index))
		d = (csvdat.loc[i, 'date']-start).days
		item_id =  csvdat.loc[i, 'item_id']
		res_qty_alipay_njhs.loc[item_id, d] = csvdat.loc[i, 'qty_alipay_njhs']
		res_pv_ipv.loc[item_id, d] = csvdat.loc[i, 'pv_ipv']
		res_pv_uv.loc[item_id, d] = csvdat.loc[i, 'pv_uv']
		res_cart_ipv.loc[item_id, d] = csvdat.loc[i, 'cart_ipv']
		res_cart_uv.loc[item_id, d] = csvdat.loc[i, 'cart_uv']
		res_collect_uv.loc[item_id, d] = csvdat.loc[i, 'collect_uv']
		res_num_gmv.loc[item_id, d] = csvdat.loc[i, 'num_gmv']
		res_qty_gmv.loc[item_id, d] = csvdat.loc[i, 'qty_gmv']
		res_unum_gmv.loc[item_id, d] = csvdat.loc[i, 'unum_gmv']

		res_num_alipay.loc[item_id, d] = csvdat.loc[i, 'num_alipay']
		res_qty_alipay.loc[item_id, d] = csvdat.loc[i, 'qty_alipay']
		res_ztc_pv_ipv.loc[item_id, d] = csvdat.loc[i, 'ztc_pv_ipv']
		res_tbk_pv_ipv.loc[item_id, d] = csvdat.loc[i, 'tbk_pv_ipv']
		res_ss_pv_ipv.loc[item_id, d] = csvdat.loc[i, 'ss_pv_ipv']
		res_jhs_pv_uv.loc[item_id, d] = csvdat.loc[i, 'jhs_pv_uv']
		res_num_alipay_njhs.loc[item_id, d] = csvdat.loc[i, 'num_alipay_njhs']
		res_unum_alipay_njhs.loc[item_id, d] = csvdat.loc[i, 'unum_alipay_njhs'] 


	res_qty_alipay_njhs.to_csv('dat/all_res_qty_alipay_njhs.csv', header = False)
	res_pv_ipv.to_csv('dat/all_res_pv_ipv.csv', header = False)
	res_pv_uv.to_csv('dat/all_res_pv_uv.csv', header = False)
	res_cart_ipv.to_csv('dat/all_res_cart_ipv.csv', header = False)
	res_cart_uv.to_csv('dat/all_res_cart_uv.csv', header = False)
	res_collect_uv.to_csv('dat/all_res_collect_uv.csv', header = False)
	res_num_gmv.to_csv('dat/all_res_num_gmv.csv', header = False)
	res_qty_gmv.to_csv('dat/all_res_qty_gmv.csv', header = False)
	res_unum_gmv.to_csv('dat/all_res_unum_gmv.csv', header = False)

	res_num_alipay.to_csv('dat/all_res_num_alipay', header = False)
	res_qty_alipay.to_csv('dat/all_res_qty_alipay', header = False)
	res_ztc_pv_ipv.to_csv('dat/all_res_ztc_pv_ipv.csv', header = False)
	res_tbk_pv_ipv.to_csv('dat/all_res_tbk_pv_ipv.csv', header = False)
	res_ss_pv_ipv.to_csv('dat/all_res_ss_pv_ipv.csv', header = False)
	res_jhs_pv_uv.to_csv('dat/all_res_jhs_pv_uv.csv', header = False)
	res_num_alipay_njhs.to_csv('dat/all_res_num_alipay_njhs.csv', header = False)
	res_unum_alipay_njhs.to_csv('dat/all_res_unum_alipay_njhs.csv', header = False)


def getstorefromcsv():
	file = 'item_store_feature2.csv'
	csvdat = pd.read_csv(file, header = None, names = [
		'date',
		'item_id',
		'store_code',
		'cate_id',
		'cate_level_id',
		'brand_id',
		'supplier_id',
		'pv_ipv',
		'pv_uv',
		'cart_ipv',
		'cart_uv',
		'collect_uv',
		'num_gmv',
		'amt_gmv',
		'qty_gmv',
		'unum_gmv',
		'amt_alipay',
		'num_alipay',
		'qty_alipay',
		'unum_alipay',
		'ztc_pv_ipv',
		'tbk_pv_ipv',
		'ss_pv_ipv',
		'jhs_pv_ipv',
		'ztc_pv_uv',
		'tbk_pv_uv',
		'ss_pv_uv',
		'jhs_pv_uv',
		'num_alipay_njhs',
		'amt_alipay_njhs',
		'qty_alipay_njhs',
		'unum_alipay_njhs'
		])
	item = getitem()
	start = pd.to_datetime('20141001')
	end = pd.to_datetime('20151227')
	res_qty_alipay_njhs = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_pv_ipv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_pv_uv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_cart_ipv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_cart_uv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_collect_uv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_num_gmv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_qty_gmv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_unum_gmv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	
	res_num_alipay = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_qty_alipay = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_ztc_pv_ipv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_tbk_pv_ipv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_ss_pv_ipv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_jhs_pv_uv = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_num_alipay_njhs = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]
	res_unum_alipay_njhs = [pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id),pd.DataFrame(-1*np.ones((len(item),(end-start).days+1)), index = item.id)]

	csvdat.date = pd.to_datetime(csvdat.date,format='%Y%m%d')
	for i in range(len(csvdat.index)):
		print i/float(len(csvdat.index))
		d = (csvdat.loc[i, 'date']-start).days
		item_id =  csvdat.loc[i, 'item_id']
		store_code = csvdat.loc[i, 'store_code']-1
		res_qty_alipay_njhs[store_code].loc[item_id, d] = csvdat.loc[i, 'qty_alipay_njhs']
		res_pv_ipv[store_code].loc[item_id, d] = csvdat.loc[i, 'pv_ipv']
		res_pv_uv[store_code].loc[item_id, d] = csvdat.loc[i, 'pv_uv']
		res_cart_ipv[store_code].loc[item_id, d] = csvdat.loc[i, 'cart_ipv']
		res_cart_uv[store_code].loc[item_id, d] = csvdat.loc[i, 'cart_uv']
		res_collect_uv[store_code].loc[item_id, d] = csvdat.loc[i, 'collect_uv']
		res_num_gmv[store_code].loc[item_id, d] = csvdat.loc[i, 'num_gmv']
		res_qty_gmv[store_code].loc[item_id, d] = csvdat.loc[i, 'qty_gmv']
		res_unum_gmv[store_code].loc[item_id, d] = csvdat.loc[i, 'unum_gmv']

		res_num_alipay[store_code].loc[item_id, d] = csvdat.loc[i, 'num_alipay']
		res_qty_alipay[store_code].loc[item_id, d] = csvdat.loc[i, 'qty_alipay']
		res_ztc_pv_ipv[store_code].loc[item_id, d] = csvdat.loc[i, 'ztc_pv_ipv']
		res_tbk_pv_ipv[store_code].loc[item_id, d] = csvdat.loc[i, 'tbk_pv_ipv']
		res_ss_pv_ipv[store_code].loc[item_id, d] = csvdat.loc[i, 'ss_pv_ipv']
		res_jhs_pv_uv[store_code].loc[item_id, d] = csvdat.loc[i, 'jhs_pv_uv']
		res_num_alipay_njhs[store_code].loc[item_id, d] = csvdat.loc[i, 'num_alipay_njhs']
		res_unum_alipay_njhs[store_code].loc[item_id, d] = csvdat.loc[i, 'unum_alipay_njhs']

	for i in range(5):
		res_qty_alipay_njhs[i].to_csv('dat/'+str(i+1)+'_res_qty_alipay_njhs.csv', header = False)
		res_pv_ipv[i].to_csv('dat/'+str(i+1)+'_res_pv_ipv.csv', header = False)
		res_pv_uv[i].to_csv('dat/'+str(i+1)+'_res_pv_uv.csv', header = False)
		res_cart_ipv[i].to_csv('dat/'+str(i+1)+'_res_cart_ipv.csv', header = False)
		res_cart_uv[i].to_csv('dat/'+str(i+1)+'_res_cart_uv.csv', header = False)
		res_collect_uv[i].to_csv('dat/'+str(i+1)+'_res_collect_uv.csv', header = False)
		res_num_gmv[i].to_csv('dat/'+str(i+1)+'_res_num_gmv.csv', header = False)
		res_qty_gmv[i].to_csv('dat/'+str(i+1)+'_res_qty_gmv.csv', header = False)
		res_unum_gmv[i].to_csv('dat/'+str(i+1)+'_res_unum_gmv.csv', header = False)

		res_num_alipay[i].to_csv('dat/'+str(i+1)+'_res_num_alipay', header = False)
		res_qty_alipay[i].to_csv('dat/'+str(i+1)+'_res_qty_alipay', header = False)
		res_ztc_pv_ipv[i].to_csv('dat/'+str(i+1)+'_res_ztc_pv_ipv.csv', header = False)
		res_tbk_pv_ipv[i].to_csv('dat/'+str(i+1)+'_res_tbk_pv_ipv.csv', header = False)
		res_ss_pv_ipv[i].to_csv('dat/'+str(i+1)+'_res_ss_pv_ipv.csv', header = False)
		res_jhs_pv_uv[i].to_csv('dat/'+str(i+1)+'_res_jhs_pv_uv.csv', header = False)
		res_num_alipay_njhs[i].to_csv('dat/'+str(i+1)+'_res_num_alipay_njhs.csv', header = False)
		res_unum_alipay_njhs[i].to_csv('dat/'+str(i+1)+'_res_unum_alipay_njhs.csv', header = False)



def mutilclean(dat):
	res = []
	for i in range(len(dat)):
		dat[i] = dat[i].drop(dat[i].index[[41,72,406,437]]).values
	# seq = dat[0]
	l = len(dat[0])
	# pdb.set_trace()
	for i in range(len(dat[0])):
		if dat[0][i] != -1:
			break;
	# pdb.set_trace()
	s1 = dat[0][i:l]
	for j in range(len(dat)):
		s = dat[j][i:l]
		s[s==-1] = 0
		res.append(s)
	# set missing data to zero
	if len(s1)<=18:
		return -1,s1
	# sl = s1[len(s1)-21:len(s1)]
	# if len(sl[sl>0])<7:
	# 	return -2,s1[len(s1)-14-1:len(s1)]
	return 0,res

def avgmutilclean(dat,avg):
	res = []
	for i in range(len(dat)):
		dat[i] = dat[i].drop(dat[i].index[[41,72,406,437]]).values
	# seq = dat[0]
	l = len(dat[0])
	# pdb.set_trace()
	for i in range(len(dat[0])):
		if dat[0][i] != -1:
			break;
	# pdb.set_trace()
	s1 = dat[0][i:l]
	s1[s1==-1] = 0
	if len(s1)<=14+avg+1:
		return -1,s1
	sl = s1[len(s1)-14:len(s1)]
	return -2,s1[len(s1)-10-1:len(s1)]
	if len(sl[sl>0])<7:
		return -2,s1[len(s1)-14-1:len(s1)]
	# set missing data to zero
	for j in range(len(dat)):
		s = dat[j][i:l]
		s[s==-1] = 0
		ag = []
		for k in range(len(s)-avg+1):
			ag.append(np.mean(s[k:k+avg]))
		res.append(ag)
	# pdb.set_trace()
	return 0,res



#dat[0] buy count
def mutilgemfeature(dat, step, ahead):
	# pdb.set_trace()
	length = len(dat) 
	seq = dat[0]
	m = len(seq)- step - ahead
	res = np.zeros((m, 1+length*step+step-1))
	index = 0;
	for i in range(m):
		y = sum(seq[i+step:i+step+ahead])
		res[index, 0] = y
		for k in range(length):
			res[index, k*step+1:(k+1)*step+1] = dat[k][i:i+step]
		for j in range(step):
			res[index, length*step+j] = dat[0][i+1] - dat[0][i]
		index += 1
	x = np.zeros((1, length*step))
	# pdb.set_trace()
	for k in range(1,length):
		x[0, k*step:(k+1)*step] = dat[k][len(dat[k])-step:len(dat[k])]
	return res,x


def mutilgetith(dat, pid):
	res = []
	# pdb.set_trace()
	for i in range(len(dat)):
		res.append(dat[i].ix[pid])
	return res

def mutilgetseq(tag):
	dat = []
	file = [
		'dat/'+tag+'_res_qty_alipay_njhs.csv',
		'dat/'+tag+'_res_unum_gmv.csv',
		'dat/'+tag+'_res_pv_ipv.csv',
		'dat/'+tag+'_res_pv_uv.csv',
		'dat/'+tag+'_res_cart_ipv.csv',
		'dat/'+tag+'_res_cart_uv.csv',
		'dat/'+tag+'_res_collect_uv.csv',
		'dat/'+tag+'_res_num_gmv.csv',
		'dat/'+tag+'_res_qty_gmv.csv',
		'dat/'+tag+'_res_num_alipay',
		'dat/'+tag+'_res_qty_alipay',
		'dat/'+tag+'_res_ztc_pv_ipv.csv',
		'dat/'+tag+'_res_tbk_pv_ipv.csv',
		'dat/'+tag+'_res_ss_pv_ipv.csv',
		'dat/'+tag+'_res_jhs_pv_uv.csv',
		'dat/'+tag+'_res_num_alipay_njhs.csv',
		'dat/'+tag+'_res_unum_alipay_njhs.csv'
		]
	for f in file:
		d = pd.read_csv(f, index_col = 0, header = None)
		dat.append(d)
	return dat


if __name__ == '__main__':
	# item = getitem()
	# getall()
	# seq = getseq('all')
	# flag,s = clean(seq.ix[104536])
	# print flag
	# print s
	# plt.plot(s)
	# plt.show()
	# _,s = clean(seq.ix[38341])
	# print s.shape,s
	# print gemfeature(s,14,14)

	# getstorebrow()
	getfromcsv()
	# getstorefromcsv()
	
	# pdb.set_trace()
	# dat = getseq('all')
	# seq = getith(dat,33796)
	# flag,s = clean(seq)
	# seq,x = gemfeature(s,7,14)
	# print seq.shape
	# print x