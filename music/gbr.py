#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from nnet_ts import *
from libnnet import *
# import pdb
import MySQLdb as mdb
from sklearn.ensemble import GradientBoostingRegressor

def readtimeseries(file):
	dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
	data = pd.read_csv(file, parse_dates = 2, names=['id', 'plays', 'date'], header= None, index_col = 2, date_parser = dateparse)
	data = data.drop(['id'], axis=1)
	return data

def querydat():
	sql = 'select * from playtimes where artist_id = "445a257964b9689f115a69e8cc5dcb75"'
	con = mdb.connect('localhost', 'sealyn', 'lyn520', 'xiami')
	dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
	with con:
		data = pd.read_sql(sql = sql, con = con)
	data = data.drop(['artist_id'], axis=1)
	npd = data.plays.values
	ts = pd.DataFrame(npd, index = pd.to_datetime(data.ds), columns=['plays'])
	return ts

if __name__ == '__main__':

	# read file
	# file = '023406156015ef87f99521f3b343f71f.csv'
	# file = '25739ad1c56a511fcac86018ac4e49bb.csv'
	file = '25739ad1c56a511fcac86018ac4e49bb.csv'
	# file = '445a257964b9689f115a69e8cc5dcb75.csv'
	# file = '5e2ef5473cbbdb335f6d51dc57845437.csv'
	data = readtimeseries(file)


	# timeseries = data.plays
	# ts = np.array(timeseries)
	# neural_net = TimeSeriesNnet(hidden_layers = [10, 5, 2], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])
	# neural_net.fit(ts, lag = 85, epochs = 10000)
	# neural_net.predict_ahead(n_ahead = 60)

	timeseries = data.plays[0:-30]
	ytest = data.plays[-31:-1]
	ts = np.array(timeseries)

	xtrain = range(len(ts))
	ytrain = ts

	print xtrain
	print ytrain

	clf = GradientBoostingRegressor().fit(xtrain, ytrain)

	xtest = range(len(ts),len(ts)+30)

	pre = clf.predict(xtest)

	# neural_net = TimeSeriesNnet(hidden_layers = [10, 5, 2], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])
	# neural_net.fit(ts, lag = 85, epochs = 10000)
	# neural_net.predict_ahead(n_ahead = 30)

	# res = np.zeros((len(ytest), 3))
	# res[:,0] = ytest
	# res[:,1] = neural_net.predictions
	# res[:,2] = neural_net.predictions-ytest
	# result = pd.DataFrame(res, index = ytest.index, columns=['trust', 'pre', 'error'])

	res = np.zeros((len(ytest), 3))
	res[:,0] = ytest
	res[:,1] = pre
	res[:,2] = pre-ytest
	result = pd.DataFrame(res, index = ytest.index, columns=['trust', 'pre', 'error'])

	print result

	# plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions', linewidth=1)
	# plt.plot(range(len(data.plays)), data.plays, '-g',  label='Original series')
	# plt.title("music data")
	# plt.xlabel("Observation ordered index")
	# plt.ylabel("No. of passengers")
	# plt.legend()
	# plt.show()


	# plt.plot()