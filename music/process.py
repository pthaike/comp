#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARMAResults
import MySQLdb as mdb
# import pdb


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

def readtimeseries(file):
	dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
	data = pd.read_csv(file, parse_dates = 2, names=['id', 'plays', 'date'], header= None, index_col = 2, date_parser = dateparse)
	data = data.drop(['id'], axis=1)
	return data

def stationarity_test(timeseries):
	rolmean = pd.rolling_mean(timeseries,  window = 12)
	rolstd = pd.rolling_std(timeseries, window = 12)

	orig = plt.plot(timeseries, color= 'red', label='Original')
	mean = plt.plot(rolmean, color = 'blue', label = 'Rolling Mean')
	std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')

	# show the figure
	plt.legend(loc='best')
	plt.title('Rolling Mean & Std')
	
	# test adfuller http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.stattools.adfuller.html
	dftest = adfuller(timeseries.plays, autolag = 'AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-values', '#Lags Used', 'Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)' % key] = value
	print dfoutput
	plt.show()


def station_pre(ts):
	ts_log = np.log(ts)
	moving_avg = pd.rolling_mean(ts_log,12)
	ts_log_moving_avg_diff = ts_log - moving_avg
	ts_log_moving_avg_diff.dropna(inplace = True)
	stationarity_test(ts_log_moving_avg_diff)

def trend(ts):
	ts_log = np.log(ts)
	ts_log_diff = ts_log - ts_log.shift()
	ts_log_diff.dropna(inplace = True)
	# stationarity_test(ts_log_diff)
	return ts_log, ts_log_diff

def decompose_pre(ts):
	ts_log = np.log(ts)
	decomposition = seasonal_decompose(ts_log.values, freq = 24)
	# decomposition.plot()
	# plt.show(block= False)
	ts_log_decompose = ts_log
	ts_log_decompose.plays = decomposition.resid
	# print ts_log_decompose
	ts_log_decompose.dropna(inplace = True)
	stationarity_test(ts_log_decompose)
	return ts_log_decompose

def acf_pacf(ts):
	ts_log, ts_log_diff = trend(ts)
	lag_acf = acf(ts_log_diff, nlags = 20)
	lag_pacf = pacf(ts_log_diff, nlags = 20, method = 'ols')

	#plot acf
	plt.subplot(121)
	plt.plot(lag_acf)
	plt.axhline(y=0, linestyle = '--', color = 'gray')
	plt.axhline(y = -1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
	plt.axhline(y = 1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
	plt.title('Autocorrelation Function')

	#plot pacf
	plt.subplot(122)
	plt.plot(lag_pacf)
	plt.axhline(y=0, linestyle = '--', color = 'gray')
	plt.axhline(y = -1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
	plt.axhline(y = 1.96/np.sqrt(len(ts_log_diff)), linestyle = '--', color = 'gray')
	plt.title('Partial Autocorrelation Function')

	plt.tight_layout()
	plt.show()

def armodel(ts):
	ts_log, ts_log_diff = trend(ts)
	model = ARIMA(ts_log, order = (1,1,0))
	result_AR = model.fit(disp = -1)
	plt.plot(ts_log_diff)
	plt.plot(result_AR.fittedvalues, color = 'red')
	# pdb.set_trace()
	plt.title('RSS: %.4F' % np.sum((result_AR.fittedvalues - ts_log_diff)**2))
	plt.show(block = False)

def mamodel(ts):
	ts_log, ts_log_diff = trend(ts)
	model = ARIMA(ts_log, order = (0,1,1))
	result_MA = model.fit(disp = -1)
	plt.plot(ts_log_diff)
	plt.plot(result_MA.fittedvalues, color = 'red')
	plt.title('RSS: %.4F' % np.sum((result_MA.fittedvalues - ts_log_diff)**2))
	plt.show(block = False)

def arimamodel(ts):
	ts_log, ts_log_diff = trend(ts)
	model = ARIMA(ts_log, order = (2,1,2))
	result_ARIMA = model.fit(disp = -1)

	m = ARIMA(ts, order = (2,1,2)).fit()

	arimares = ARMAResults(m, params = '')

	pre = arimares.forcast(steps = 60)


	# pre = m.predict('20150901', '20151230', dynamic = True)
	print pre

	# prediction back to the original scale
	predictions_ARIMA = backorg(result_ARIMA, ts_log)
	plt.plot(predictions_ARIMA)
	# print (predictions_ARIMA - ts)[40:80]

	plt.plot(ts, color = 'red')

	# plt.plot(ts_log_diff)
	# plt.plot(result_ARIMA.fittedvalues, color = 'red')
	plt.title('RSS: %.4F' % np.sum((result_ARIMA.fittedvalues - ts_log_diff)**2))
	plt.show()

	


def backorg(pre, ts_log):
	# pdb.set_trace()
	predictions_ARIMA_diff = pd.Series(pre.fittedvalues, copy = True)
	predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
	predictions_ARIMA_log = pd.Series(ts_log.plays,index = ts_log.index)
	predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value = 0)
	predictions_ARIMA = np.exp(predictions_ARIMA_log)

	return predictions_ARIMA


if __name__ == '__main__':

	# read file
	# file = '023406156015ef87f99521f3b343f71f.csv'
	# file = '25739ad1c56a511fcac86018ac4e49bb.csv'
	# file = '25739ad1c56a511fcac86018ac4e49bb.csv'
	file = 'allplay/play/445a257964b9689f115a69e8cc5dcb75.csv'
	# # file = '5e2ef5473cbbdb335f6d51dc57845437.csv'
	data = readtimeseries(file)

	# #acf and pacf
	# # acf_pacf(data)

	# arma_mod = ARMA(data, order= (2,2))
	# arma_res = arma_mod.fit(trend = 'nc', disp = -1)
	# # arma_res.plot_predict(start = '2015-05-01', end = '2015-08-03')
	# re = arma_res.predict(start = '2015-09-01', end = '2015-12-03')
	# re.plot
	# plt.show()


	stationarity_test(data)

	


	
	# data = querydat()
	# data = trend(data)


	# station_pre(data)
	# data = trend(data)
	# stationarity_test(data)
	# acf_pacf(data)

	# arimamodel(data)