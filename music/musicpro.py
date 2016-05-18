#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

def getartist():
	file = 'art.csv'
	name = pd.read_csv(file, header=None, names = ['id','num'])
	return name

def getsonginfo():
	file = 'mars_tianchi_songs.csv'
	dat = pd.read_csv(file, header = None, index_col = 0, names = [
		'song_id',
		'artist_id',
		'publish_time',
		'song_init_plays',
		'Language',
		'Gender'
		])
	return dat

def prepare():
	file = 'mars_tianchi_user_actions.csv'
	songinfo = getsonginfo()
	csvdat = pd.read_csv(file, header = None, names = [
		'user_id',
		'song_id',
		'gmt_create',
		'action_type',
		'Ds'
		])
	d = csvdat[csvdat]

def getdat(aid):
	file = ['allplay/play/'+aid+'.csv', 'allplay/collect/'+aid+'.csv', 'allplay/download/'+aid+'.csv']
	start = pd.to_datetime('20150301')
	end = pd.to_datetime('20150830')
	res = np.zeros(((end-start).days+1, len(file)))
	for t in range(len(file)):
		d = pd.read_csv(file[t])
		d.ds = pd.to_datetime(d.ds, format='%Y%m%d')
		for i in d.index:
			dc = (d.loc[i, 'ds']-start).days
			res[dc, t] = d.loc[i, 'down']
	return res

def genfeature(ts, step):
	n = len(ts)
	x = np.zeros((n-step, step))
	y = np.zeros((n-step, 1))
	for i in range(n-step):
		x[i, :] = ts[i:i+step]
		y[i, 0] = ts[i+step]
	# pdb.set_trace()
	x_pre = ts[n-step:n]
	return x,y,x_pre

'''
add the feature of download and collect
step: ts before step size
ostep: down and collect step size
'''
def genmutilfeature(ts, down, collect, step, ostep):
	n = len(ts)
	ndwon = len(down)
	ncoll = len(collect)
	x = np.zeros((n-step, step + 2 * ostep))
	y = np.zeros((n-step, 1))
	for i in range(n-step):
		x[i, 0:step] = ts[i:i+step]
		x[i, step:step+ostep] = down[i:i+ostep]
		x[i, step+ostep:step+2*ostep] = collect[i:i+ostep]
		y[i, 0] = ts[i+step]
	# pdb.set_trace()
	x_pre = ts[n-step:n]
	x_pre = np.concatenate((x_pre, down[ndwon-ostep:ndwon], collect[ncoll-ostep:ncoll]), axis = 0)
	return x, y, x_pre


if __name__ == '__main__':
	# getsonginfo()
	# print getartist()
	
	d = getdat('3e395c6b799d3d8cb7cd501b4503b536')
	x,y,x_pre = genfeature(d[:,0], 14)
	print x,y,x_pre
