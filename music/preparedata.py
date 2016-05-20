#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import MySQLdb as mdb
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pdb

# get artist list
def getartist():
	file = 'art.csv'
	name = pd.read_csv(file, header=None, names = ['id','num'])
	return name

#save the data to local file
def predat(table, artist):
	con = mdb.connect('localhost', 'sealyn', 'lyn520', 'xiami')
	for id in artist.id:
		sql = 'select * from '+table+' where artist_id = "' + id + '"'
		with con:
			print table+id
			data = pd.read_sql(sql = sql, con = con)
			data.to_csv(table+'/'+id+'.csv')

# topk plays everyday
def statsplay(topk):
	artistlist = getartist()
	daterange = pd.period_range('3/1/2015', '8/30/2015', freq='D')
	dat = np.zeros((len(daterange), topk))
	for id in artistlist.id:
		artist_id = id
		print id
		for i in range(len(daterange)):
			date = str(daterange[i])
			sql = 'select songs.song_id as song_is, count(*) as plays from actions, songs where action_type = 1 and ds = "'+date+'" and songs.artist_id = "'+artist_id+'" and actions.song_id = songs.song_id group by songs.song_id order by count(*) desc limit '+ str(topk)
			con = mdb.connect('localhost', 'sealyn', 'lyn520', 'xiami')
			with con:
				data = pd.read_sql(sql = sql, con = con)
				if(data.shape[0]<topk):
					dat[i, 0:data.shape[0]] = data.plays
				else:
					dat[i, :] = data.plays
		df = pd.DataFrame(dat)
		df.to_csv('topk/play/'+id+'.csv', header = False, index = False)


""" 
preday:(int) the count of days before we should consider
type:(string) play, download, collect
artist_id : (string)
"""
def gempredat(preday, type, artist_id):

	playfile = 'allplay/'+type+'/'+artist_id+'.csv'
	dfplay = pd.read_csv(playfile)
	dat_x = np.zeros((dfplay.shape[0]-preday, preday))
	dat_y = np.zeros((dfplay.shape[0]-preday, 1))
	dat = np.zeros((dfplay.shape[0]-preday, preday+1))
	# print dfplay.down[0: preday]
	# print dfplay.down[preday]
	for d in range(dfplay.shape[0]-preday):
		dat_x[d,] = dfplay.down[d: d+preday]
		dat_y[d] = dfplay.down[d+preday]
	# dat[:,0] = dat_y[:,0]
	# dat[:,1:preday+1] = dat_x
	return dat_x, dat_y



def predict(clf, x_pre, ahead):
	pre = np.zeros((ahead,1))
	# pdb.set_trace()
	x = x_pre
	for i in range(ahead):
		p = clf.predict(x)
		print p
		pre[i] = p
		x[0:-1] = x_pre[1:len(x_pre)]
		x[-1] = p
	return pre



if __name__ == '__main__':
	# artist = getartist()
	# # table = 'play'
	# table = 'download'
	# predat(table, artist)
	# table = 'collect'
	# predat(table, artist)
	statsplay(5)

	# preday = 30
	# ahead = 30
	# x, y = gempredat(preday, 'play','ffd47cf9cb66d226575336f0fa42ae25')

	# xtrain = x[0:120,:]
	# ytrain = y[0:120,0]
	# xtest = x[120:120+ahead, :]
	# ytest = y[120:120+ahead]
	# print xtest.shape
	# print ytest.shape


	# clf = GradientBoostingRegressor(n_estimators=10,max_leaf_nodes =20, learning_rate=0.1,max_depth=3, random_state=100, loss='ls').fit(xtrain, ytrain)
	# # pre = clf.predict(xtrain)

	# pre = predict(clf, xtest[0,:], ahead)
	# print pre
	# print np.mean(abs(pre - ytrain))

	# print ytrain


	# print y
	# xtrain = x[0:120, :]
	# ytrain = y[0:120]
	# print ytrain.shape
	# xtest = x[120:x.shape[0], :]
	# ytest = y[120:x.shape[0]]
	# ahead = x.shape[0]-120
	# clf = GradientBoostingRegressor(n_estimators=10,max_leaf_nodes =20, learning_rate=0.1,max_depth=3, random_state=100, loss='ls').fit(xtrain, ytrain)
	# # print xtest
	# x_pre = xtest[0,:]
	# pre = predict(clf, x_pre, ahead)
	# print pre
	# res = np.zeros((len(ytest), 3))
	# res[:,0] = ytest[:,0]
	# res[:,1] = pre
	# print  ytest[:,0].shape
	# print len(pre)
	# res[:,2] = ytest[:,0] -pre

	# df = pd.DataFrame(res, columns=['trust', 'pre', 'error'])
	# print df
	# a = abs(df.error)
	# print np.mean(a)
