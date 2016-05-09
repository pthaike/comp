#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
from sklearn.ensemble.mygradient import MyGradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pdb
from sklearn import grid_search
import xgboost as xgb
from birdpro import *
import time
from sklearn.svm import SVR
# from libnnet import *

submit = True

def loss_func(ground_truth, prediction):
	pass

def splitdat(x,y):
	indices = np.arange(y.shape[0])
	np.random.shuffle(indices)
	x, y = x[indices], y[indices]
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
	return xtrain, xtest, ytrain, ytest


def printscore(pre, y):
	a = abs(pre-y)
	res = np.zeros((len(y),3)) 
	res[:,0] = y
	res[:,1] = pre
	res[:,2] = pre - y
	stat = pd.DataFrame(res, columns=['y', 'pre', 'error'])
	print stat
	print np.mean(abs(stat.error))


def train(x, y, weight):
	if submit:
		xtrain = x
		ytrain = y
	else:
		xtrain, xtest, ytrain, ytest = splitdat(x, y)
	clf = MyGradientBoostingRegressor(n_estimators=80,max_leaf_nodes =20, learning_rate=0.1,max_depth=3, random_state=300, loss='my', cost_weight = weight).fit(xtrain, ytrain)
	if not submit:
		pre = clf.predict(xtest)
		printscore(pre, ytest)
	return clf

def logreboj(preds, dtrain):
	lables = dtrain.get_label()

def evalerror(preds, dtrain, cost_weight):
	# pdb.set_trace()
	lables = dtrain.get_label()
	error = lables-preds
	pos = abs(error[error>0])
	neg = abs(error[error<0])
	c = (np.sum(cost_weight[0]*pos) + np.sum(cost_weight[1]*neg))
	return 'error', float(c/len(error))

def svrtrain(x,y, pre_x):
	clf = SVR(C=1.0, epsilon=0.2, kernel = 'linear')
	clf.fit(x,y)
	pred = clf.predict(pre_x)
	return pred

def gbdttrain(x, y, pre_x, weight):
	clf = MyGradientBoostingRegressor(n_estimators=100,max_leaf_nodes =10, learning_rate=0.1,max_depth=6, random_state=400, loss='ls').fit(x, y)
	# clf = MyGradientBoostingRegressor(n_estimators=80,max_leaf_nodes =20, learning_rate=0.1,max_depth=3, random_state=300, loss='my', cost_weight = weight).fit(x, y)

	pred = clf.predict(pre_x)
	return pred


def xgbtrain(x,y, pre_x, cost_weight):
	# xtrain, xtest, ytrain, ytest = splitdat(x, y)
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
	# xtrain, ytrain = x, y
	# pdb.set_trace()

	dtrain = xgb.DMatrix(xtrain, label = ytrain, missing = -1)
	dvalid = xgb.DMatrix(xtest, label = ytest, missing = -1)
	# print pre_x
	# pre_x = np.array([pre_x, pre_x])
	dpre_x = xgb.DMatrix(pre_x, missing = -1)
	dprex = xgb.DMatrix(x, missing = -1)
	
	param = {
		'booster':'gbtree',
		'objective':'reg:linear',
		'early_stopping_rounds':500,
		'max_depth':6,
		'subsample':0.7,
		'silent' : 1,
		'colssample_bytree':0.8,
		'eta':0.02,
		'nthread':10,
		'seed':400
	}
	watchlist = [ (dtrain,'train'), (dvalid,'val')]
	# model = xgb.train(param, dtrain, num_boost_round=1000, evals=watchlist)
	model = xgb.train(param, dtrain, num_boost_round = 2000, evals = watchlist, feval = evalerror, cost_weight = cost_weight)

	# model.save_model('xgb.model')
	print 'predict....'
	#predict
	pre_y = model.predict(dpre_x, ntree_limit=model.best_iteration)
	prey = model.predict(dprex, ntree_limit=model.best_iteration)
	# printscore(ytest,pre_y)
	return pre_y,prey



def voting(x,y):
	xtrain, xtest, ytrain, ytest = splitdat(x, y)
	algorithms = [GradientBoostingRegressor, RandomForestRegressor]
	predictions = matrix(len(ytest), len(algorithms))
	for i, algorithm in enumerate(algorithms):
		predictions[:,i] = algorithm.fit(xtrain)

# the version of predict all 14 days together
def aheadpredict(step, ahead, tag, weight):
	seq = getseq(tag)
	res = np.zeros(seq.shape[0])
	index = 0
	for s in seq.index:
		product = s
		flag,ts = clean(seq.ix[s])
		w = [1,1]
		w[0] = weight[product][tag]['pos']
		w[1] = weight[product][tag]['neg']
		pred = 0
		if flag == 0:
			# pdb.set_trace()
			predictions = np.zeros(ahead)
			f,pre_x = gemfeature(ts, step, ahead)
			y = f[:, 0]         
			x = f[:, 1:f.shape[1]]
			pre = xgbtrain(x, y, pre_x)
			pred = pre[0]
		else:
			a = len(ts[ts>0])
			if a == 0:
				pred = 0;
			else:
				pred = sum(ts)/a*ahead;
		res[index] = pred
		index += 1
		# break
		print product,pred
	dat = pd.DataFrame([tag]*seq.shape[0], index = seq.index, columns = ['type'])
	dat['pred'] = res
	return dat
def submit():
	step = 7
	ahead = 14
	weight = getweight()
	sub = aheadpredict(step,ahead,'all', weight)
	for i in range(1,6):
		d = aheadpredict(step,ahead,str(i),weight)
		sub = sub.append(d)
	now = time.strftime('%Y%m%d%H%M%S')
	print sub
	sub.pred[sub.pred<0] = 0
	sub.to_csv('res/'+now+'xgb7_14.csv', header=False)

# mutil features prediction
def aheadmutilpredict(step, ahead, tag, item, cost_weight):
	seq = mutilgetseq(tag)
	res = np.zeros(len(item))
	file1 = open(tag+'_data_y.csv','w')
	file2 = open(tag+'_pre_res.csv','w')
	index = 0
	for pid in item.id:
		s = mutilgetith(seq, pid)
		flag,ts = avgmutilclean(s,2)
		w = [1,1]
		w[0] = cost_weight[pid][tag]['pos']
		w[1] = cost_weight[pid][tag]['neg']
		pred = 0
		if flag == 0:
			
			f,pre_x = mutilgemfeature(ts, step, ahead)
			y = f[:, 0]
			x = f[:, 1:f.shape[1]]
			pre_y, prey = xgbtrain(x, y, pre_x, w)
			# pdb.set_trace()
			file1.write(str(pid))
			file1.write(',')
			for i in prey:
				file1.write(str(i))
				file1.write(',')
			file1.write(pre_y)
			file1.write('\n')
		else:
			# pdb.set_trace()
			file2.write(str(pid))
			file2.write(',')
			a = len(ts)
			if a == 0:
				pred = 0;
			else:
				pred = sum(ts)/a*ahead;
			file2.write(tag)
			file2.write(',')
			file2.write(str(pred))
			file2.write('\n')
		# res[index] = pred
		index += 1
		print pid,pred,w
	file2.close()
	file1.close()
		# break
		# pdb.set_trace()
	# dat = pd.DataFrame([tag]*len(item), index = item.id, columns = ['type'])
	# dat['pred'] = res
	# return dat


def mutilsubmit():
	step = 14
	ahead = 14
	item = getitem()
	cost_weight = getweight()
	aheadmutilpredict(step,ahead,'all', item, cost_weight)
	for i in range(1,6):
		aheadmutilpredict(step,ahead,str(i), item, cost_weight)
	# now = time.strftime('%Y%m%d%H%M%S')
	# print sub
	# sub.pred[sub.pred<0] = 0
	# sub.to_csv('res/'+now+'xgb_mutil_have28_'+str(step)+'_'+str(ahead)+'.csv', header=False)
	# print sub



def gbrtaheadmutilpredict(step, ahead, tag, item, cost_weight):
	seq = mutilgetseq(tag)
	res = np.zeros(len(item))
	index = 0
	for pid in item.id:
		s = mutilgetith(seq, pid)
		flag,ts = avgmutilclean(s,2)
		w = [1,1]
		w[0] = cost_weight[pid][tag]['pos']
		w[1] = cost_weight[pid][tag]['neg']
		pred = 0
		if flag == 0:
			# pdb.set_trace()
			f,pre_x = mutilgemfeature(ts, step, ahead)
			y = f[:, 0]
			x = f[:, 1:f.shape[1]]
			pre = gbdttrain(x, y, pre_x, w)
			pred = pre[0]
		else:
			# pdb.set_trace()
			a = len(ts)
			if a == 0:
				pred = 0;
			else:
				pred = sum(ts)/a*ahead;
		res[index] = pred
		index += 1
		print pid,pred,w
		# break
		# pdb.set_trace()
	dat = pd.DataFrame([tag]*len(item), index = item.id, columns = ['type'])
	dat['pred'] = res
	return dat


def gbrtmutilsubmit():
	step = 14
	ahead = 14
	item = getitem()
	cost_weight = getweight()
	sub = gbrtaheadmutilpredict(step,ahead,'all', item, cost_weight)
	for i in range(1,6):
		d = gbrtaheadmutilpredict(step,ahead,str(i), item, cost_weight)
		sub = sub.append(d)
	now = time.strftime('%Y%m%d%H%M%S')
	print sub
	sub.pred[sub.pred<0] = 0
	sub.to_csv('res/'+now+'gbrt_mutil_'+str(step)+'_'+str(ahead)+'.csv', header=False)
	print sub


if __name__ == '__main__':
	# pred = seqpredict(14,14)
	# submit()
	mutilsubmit()
	# nnetsubmit()
	# gbrtmutilsubmit()