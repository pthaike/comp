#! /usr/bin/python

import pandas as pd 
import numpy as np


def readorder(date):
	file = 'season_1/training_data/order_data/order_data_'+date
	df = pd.read_csv(file, sep = '\t', header = None, names = ['order_id', 'driver_id', 'passenger_id', 'start_district_hash', 'dest_district_hash', 'Price', 'Time'])
	return df

def readdistric():
	file = 'season_1/training_data/cluster_map/cluster_map'
	df = pd.read_csv(file, index_col = 0, sep = '\t', header = None, names = ['district_id'])
	return df

def statorder():
	date = '2016-01-01'
	distric = readdistric()
	date = pd.period_range('2016-01-01', '2016-01-21', freq='D')
	for d in date:
		sd = str(d)
		print sd
		res = [pd.DataFrame(np.zeros((144, 2)), columns = ['yes', 'no']) for i in range(distric.shape[0])]
		order = readorder(sd)
		for i in order.index:
			t = pd.to_datetime(order.loc[i, 'Time'])
			h = t.hour
			m = t.minute
			s = t.second
			inx = (h*3600 + m * 60 + s) / 600
			did = distric.ix[order.loc[i, 'start_district_hash']].district_id-1
			if pd.isnull(order.loc[i, 'driver_id']):
				res[did].loc[inx, 'no'] += 1
			else:
				res[did].loc[inx, 'yes'] += 1
		for i,r in enumerate(res):
			r.to_csv('dat/'+sd+'-'+str(i+1), header = False, index = False)
	return res

def readdat(file):
	df = pd.read_csv('dat/'+file, header = None, names = ['yes', 'no'])
	return df

def readpredict():
	df = pd.read_csv('season_1/test_set_1/read_me_1.txt', header = None, names = ['date'])
	return df

if __name__ == '__main__':

	# df = readorder()
	# for o in df
	# print df.Time
	# print type(df.driver_id[0])
	# print df[df.driver_id.isnull()]
	# df = readdistric()
	# print df.shape[0]
	# print df.ix['693a21b16653871bbd455403da5412b4'].district_id

	print statorder()