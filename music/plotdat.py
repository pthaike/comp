#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(file):
	pdat = pd.read_csv(file)
	x = pdat.values[:,1]
	m = x.shape[0]
	y = range(m)
	print len(x)
	print len(y)
	plt.plot(y,x)
	plt.show()

def plotall(file):
	dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
	data = pd.read_csv(file, parse_dates = 0, names=['date', 'plays'], header= None, index_col = 0, date_parser = dateparse)
	data.plot()
	plt.show()

def plotbytime(file):
	dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
	data = pd.read_csv(file, parse_dates = 2, names=['id', 'plays', 'date'], header= None, index_col = 2, date_parser = dateparse)
	data.drop(['id'], axis=1)
	data.plot()
	plt.show()


if __name__ == '__main__':
	# file = '023406156015ef87f99521f3b343f71f.csv'
	file = '25739ad1c56a511fcac86018ac4e49bb.csv'
	# file = '25739ad1c56a511fcac86018ac4e49bb.csv'
	# file = '445a257964b9689f115a69e8cc5dcb75.csv'
	# file = '5e2ef5473cbbdb335f6d51dc57845437.csv'
	
	
	
	plotbytime(file)

	# file = 'allplay.csv'
	# plotall(file)