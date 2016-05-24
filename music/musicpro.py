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

def gettopk(aid):
	file = 'topk/play/' + aid + '.csv'
	df = pd.read_csv(file, header = False)
	return df.values


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

#0521
# def genmutilfeaturemore(ts, down, collect, step, ostep):
# 	n = len(ts)
# 	fnum = 3
# 	ndwon = len(down)
# 	ncoll = len(collect)
# 	x = np.zeros((n-step, step*2+fnum-1))
# 	y = np.zeros((n-step, 1))
# 	for i in range(n-step):
# 		tend = np.diff(ts[i:i+step])
# 		a = np.concatenate((ts[i:i+step], [sum(ts[i:i+ostep/2])/float(sum(ts[i+ostep/2:i+ostep]))], [np.mean(ts[i:i+ostep])], [np.var(ts[i:i+ostep])], tend), axis = 0)
# 		# x[i, 0:step] = ts[i:i+step]

# 		# x[i, step+2*ostep] = sum(ts[i:i+ostep/2])/float(sum(ts[i+ostep/2:i+ostep]))
# 		# x[i, step+2*ostep+1] = np.mean(ts[i:i+ostep])
# 		# x[i, step+2*ostep+2] = np.var(ts[i:i+ostep])
# 		x[i] = a
# 		y[i, 0] = ts[i+step]
# 	# pdb.set_trace()
# 	x_pre = np.concatenate(( ts[n-step:n], [sum(ts[n-step:n-step/2])/float(sum(ts[n-step/2:n]))], [np.mean(ts[n-step:n])], [np.var(ts[n-step:n])], np.diff(ts[n-step:n])), axis = 0)
# 	return x, y, x_pre

'''
add the feature of download and collect
step: ts before step size
ostep: down and collect step size
'''
def genmutilfeaturemore(ts, down, collect, step, ostep):
	n = len(ts)
	fnum = 3
	ndwon = len(down)
	ncoll = len(collect)
	x = np.zeros((n-step, step + 2 * ostep+fnum))
	y = np.zeros((n-step, 1))
	for i in range(n-step):
		x[i, 0:step] = ts[i:i+step]
		x[i, step:step+ostep] = down[i:i+ostep]
		x[i, step+ostep:step+2*ostep] = collect[i:i+ostep]

		x[i, step+2*ostep] = sum(ts[i:i+ostep/2])/float(sum(ts[i+ostep/2:i+ostep]))
		x[i, step+2*ostep+1] = np.mean(ts[i:i+ostep])
		x[i, step+2*ostep+2] = np.var(ts[i:i+ostep])

		y[i, 0] = ts[i+step]
	# pdb.set_trace()
	x_pre = ts[n-step:n]
	x_pre = np.concatenate((x_pre, down[ndwon-ostep:ndwon], collect[ncoll-ostep:ncoll],[sum(ts[n-step:n-step/2])/float(sum(ts[n-step/2:n]))], [np.mean(ts[n-step:n])], [np.var(ts[n-step:n])]), axis = 0)
	return x, y, x_pre





# def trend(ts):

# 	ts_log = np.log(ts)
# 	ts_log_diff = ts_log - np.shift(ts_log)
# 	# stationarity_test(ts_log_diff)
# 	return ts_log, ts_log_diff

def genmutilfeaturemoretopk(ts, down, collect, topk, step, ostep):
	n = len(ts)
	fnum = 3
	ndwon = len(down)
	ncoll = len(collect)
	tm, tn = topk.shape
	x = np.zeros((n-step, 2*step+2*ostep+fnum+ostep-1))
	y = np.zeros((n-step, 1))
	for i in range(n-step):
		x[i, 0:step] = ts[i:i+step]
		x[i, step:step+ostep] = down[i:i+ostep]
		x[i, step+ostep:step+2*ostep] = collect[i:i+ostep]

		x[i, step+2*ostep] = sum(ts[i:i+ostep/2])/float(sum(ts[i+ostep/2:i+ostep]))
		x[i, step+2*ostep+1] = np.mean(ts[i:i+ostep])
		x[i, step+2*ostep+2] = np.var(ts[i:i+ostep])

		tk = np.sum(topk[i:i+ostep], axis = 1)
		x[i, step + 2 * ostep+fnum : step+3*ostep+fnum] = tk
		# pdb.set_trace()
		tend = np.diff(ts[i:i+step])
		x[1, step+3*ostep+fnum: 2*step + 2*step+3*ostep+fnum-1] = tend


		y[i, 0] = ts[i+step]
	# pdb.set_trace()
	x_pre = ts[n-step:n]
	tk = np.sum(topk[tm-ostep:tm], axis = 1)
	x_pre = np.concatenate((x_pre, down[ndwon-ostep:ndwon], collect[ncoll-ostep:ncoll], [sum(ts[n-step:n-step/2])/float(sum(ts[n-step/2:n]))], [np.mean(ts[n-step:n])], [np.var(ts[n-step:n])], tk, np.diff(ts[n-step:n])), axis = 0)
	return x, y, x_pre


# def genmutilfeaturemoretopk(ts, down, collect, topk, step, ostep):
# 	n = len(ts)
# 	fnum = 3
# 	ndwon = len(down)
# 	ncoll = len(collect)
# 	tm, tn = topk.shape
# 	x = np.zeros((n-step, step+2*ostep+fnum+ostep))
# 	y = np.zeros((n-step, 1))
# 	for i in range(n-step):
# 		x[i, 0:step] = ts[i:i+step]
# 		x[i, step:step+ostep] = down[i:i+ostep]
# 		x[i, step+ostep:step+2*ostep] = collect[i:i+ostep]

# 		x[i, step+2*ostep] = sum(ts[i:i+ostep/2])/float(sum(ts[i+ostep/2:i+ostep]))
# 		x[i, step+2*ostep+1] = np.mean(ts[i:i+ostep])
# 		x[i, step+2*ostep+2] = np.var(ts[i:i+ostep])

# 		tk = np.sum(topk[i:i+ostep], axis = 1)
# 		x[i, step + 2 * ostep+fnum : step+3*ostep+fnum] = tk
# 		# x[i, step + 2 * ostep+fnum : step+2*ostep+fnum + tn] = topk[i+ostep-3]
# 		# x[i, step+2*ostep+fnum+tn : step+2*ostep+fnum+2*tn] = topk[i+ostep-2]
# 		y[i, 0] = ts[i+step]
# 	# pdb.set_trace()
# 	x_pre = ts[n-step:n]
# 	tk = np.sum(topk[tm-ostep:tm], axis = 1)
# 	x_pre = np.concatenate((x_pre, down[ndwon-ostep:ndwon], collect[ncoll-ostep:ncoll], [sum(ts[n-step:n-step/2])/float(sum(ts[n-step/2:n]))], [np.mean(ts[n-step:n])], [np.var(ts[n-step:n])], tk), axis = 0)
# 	return x, y, x_pre

# def genmutilfeaturemoretopk(ts, down, collect, topk, step, ostep):
# 	n = len(ts)
# 	fnum = 3
# 	ndwon = len(down)
# 	ncoll = len(collect)
# 	tm, tn = topk.shape
# 	x = np.zeros((n-step, step+2*ostep+fnum+tn*2))
# 	y = np.zeros((n-step, 1))
# 	for i in range(n-step):
# 		x[i, 0:step] = ts[i:i+step]
# 		x[i, step:step+ostep] = down[i:i+ostep]
# 		x[i, step+ostep:step+2*ostep] = collect[i:i+ostep]

# 		x[i, step+2*ostep] = sum(ts[i:i+ostep/2])/float(sum(ts[i+ostep/2:i+ostep]))
# 		x[i, step+2*ostep+1] = np.mean(ts[i:i+ostep])
# 		x[i, step+2*ostep+2] = np.var(ts[i:i+ostep])

# 		x[i, step + 2 * ostep+fnum : step+2*ostep+fnum + tn] = topk[i+ostep-3]
# 		x[i, step+2*ostep+fnum+tn : step+2*ostep+fnum+2*tn] = topk[i+ostep-2]
# 		y[i, 0] = ts[i+step]
# 	# pdb.set_trace()
# 	x_pre = ts[n-step:n]
# 	x_pre = np.concatenate((x_pre, down[ndwon-ostep:ndwon], collect[ncoll-ostep:ncoll], [sum(ts[n-step:n-step/2])/float(sum(ts[n-step/2:n]))], topk[tm-2], topk[tm-1]), axis = 0)
# 	return x, y, x_pre



def plotdat():
	art = getartist()
	daterange = pd.period_range('20150301', '20150830', freq='D')
	date = [d for d in daterange]
	for aid in art.id:
		# file = 'allplay/play/'+aid+'.csv'
		# d = pd.read_csv(file)
		d = getdat(aid)
		df = pd.DataFrame(d, index = date, columns = ['play', 'collect', 'download'])
		plt.figure
		df.plot()
		plt.title(aid)
		print aid
		plt.show()

if __name__ == '__main__':
	# getsonginfo()
	# print getartist()
	aid = '3e395c6b799d3d8cb7cd501b4503b536'
	# d = getdat(aid)
	# x,y,x_pre = genfeature(d[:,0], 14)
	# print x,y,x_pre

	# plotdat()

	print gettopk(aid)
