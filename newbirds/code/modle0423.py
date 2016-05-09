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
from libnnet import *

submit = True

def loss_func(ground_truth, prediction):
	pass


def paramselect(x,y):
	#gridsearch for param
	#test estimator
	# param_test1 = {'n_estimators':range(60,90,5)}
	# gsearch1 = grid_search.GridSearchCV(estimator=
	# 	GradientBoostingClassifier(learning_rate=0.1,min_samples_split=500,
	# 		min_samples_leaf=50,max_depth=8,max_features='sqrt',
	# 		subsample=0.8,random_state=10),
	# 	param_grid=param_test1, scoring='roc_auc',iid=False,cv=5)
	# clf = gsearch1

	# test param
	# param_test2 = {'max_depth':range(3,8,1), 'min_samples_split':range(150,301,50)}
	# gsearch2 = grid_search.GridSearchCV(estimator = 
	# 	GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 50, max_features = 'sqrt', subsample=0.8, random_state=10),
	# 	param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
	# clf = gsearch2

	#test min_samples_leaf
	# param_test3 = {'min_samples_leaf':range(191,210,5)}
	# gsearch3 = grid_search.GridSearchCV(estimator = 
	# 	GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 50, max_depth=6, min_samples_split=350, max_features = 'sqrt', subsample=0.8, random_state=10),
	# 	param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
	# clf = gsearch3

	# param_test4 = {'max_features':range(60,101,10)}
	# gsearch4 = grid_search.GridSearchCV(estimator = 
	# 	GradientBoostingClassifier(learning_rate = 0.1, min_samples_leaf = 200, n_estimators = 50, max_depth=6, min_samples_split=350, subsample=0.8, random_state=10),
	# 	param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
	# clf = gsearch4

	param_test5 = {'subsample':[0.9,1,1.2]}
	gsearch5 = grid_search.GridSearchCV(estimator = 
		GradientBoostingClassifier(learning_rate = 0.1, min_samples_leaf = 200, n_estimators = 60, max_depth=3, min_samples_split=350, random_state=10),
		param_grid=param_test5, scoring='roc_auc', iid=False, cv=5)
	clf = gsearch5
	clf = clf.fit(tx12000, ty12000)
	print clf.grid_scores_,'\n', clf.best_params_, clf.best_score_


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


def xgbtrain(x,y, pre_x):
	# xtrain, xtest, ytrain, ytest = splitdat(x, y)
	xtrain, ytrain = x, y
	# pdb.set_trace()
	dtrain = xgb.DMatrix(xtrain, label = ytrain,missing = -1)
	# print pre_x
	# pre_x = np.array([pre_x, pre_x])
	dvalid = xgb.DMatrix(pre_x, missing = -1)
	
	param = {
		'booster':'gbtree',
		'objective':'reg:linear',
		'early_stopping_rounds':100,
		'max_depth':6,
		'silent' : 1,
		'subsample':0.6,
		'colssample_bytree':0.5,
		'eta':0.02,
		'nthread':10,
		'seed':430
	}
	#2016041919
	# param = {
	# 	'booster':'gbtree',
	# 	'objective':'reg:logistic',
	# 	'early_stopping_rounds':100,
	# 	'max_depth':6,
	# 	'silent' : 1,
	# 	'subsample':0.6,
	# 	'colssample_bytree':0.5,
	# 	'eta':0.02,
	# 	'nthread':10,
	# 	'seed':430
	# }
	watchlist = [ (dtrain,'train') ]
	model = xgb.train(param, dtrain, num_boost_round=500, evals=watchlist)
	# model.save_model('xgb.model')
	print 'predict....'
	#predict
	pre_y = model.predict(dvalid,ntree_limit=model.best_iteration)
	# printscore(ytest,pre_y)
	return pre_y



def voting(x,y):
	xtrain, xtest, ytrain, ytest = splitdat(x, y)
	algorithms = [GradientBoostingRegressor, RandomForestRegressor]
	predictions = matrix(len(ytest), len(algorithms))
	for i, algorithm in enumerate(algorithms):
		predictions[:,i] = algorithm.fit(xtrain)

#first version with everyday predict
# def seqpredict(step, ahead, tag, weight):
# 	seq = getseq(tag)
# 	res = np.zeros(seq.shape[0])
# 	index = 0
# 	for s in seq.index:
# 		product = s
# 		flag,ts = clean(seq.ix[s])
# 		w = [1,1]
# 		w[0] = weight[product][tag]['pos']
# 		w[1] = weight[product][tag]['neg']
# 		pred = 0
# 		if flag == 0:
# 			# pdb.set_trace()
# 			predictions = np.zeros(ahead)
# 			f,pre_x = gemfeature(ts, step, 1)
# 			pre_x = pre_x.values
# 			for i in range(ahead):
# 				f,_ = gemfeature(ts, step+i, 1)
# 				if product == 38341:
# 					print f
# 				y = f[:, 0]
# 				x = f[:, 1:f.shape[1]]
# 				pre_x = np.array([pre_x])
# 				pre = xgbtrain(x, y, pre_x)
# 				# m = train(x,y,w)
# 				# pre = m.predict(pre_x)
# 				predictions[i] =  pre
# 				pre_x = np.concatenate((pre_x[0],pre), axis = 0)
# 				ts = np.concatenate((ts,pre), axis = 0)
# 			pred = sum(predictions)
# 		else:
# 			a = len(ts)
# 			if a == 0:
# 				pred = 0;
# 			else:
# 				pred = sum(ts)/a*ahead;
# 		res[index] = pred
# 		index += 1
# 		# break
# 		print product,pred
# 	dat = pd.DataFrame([tag]*seq.shape[0], index = seq.index, columns = ['type'])
# 	dat['pred'] = res
# 	return dat

# def submit():
# 	step = 14
# 	ahead = 14
# 	weight = getweight()
# 	sub = seqpredict(step,ahead,'all', weight)
# 	for i in range(1,6):
# 		d = seqpredict(step,ahead,str(i),weight)
# 		sub = sub.append(d)
# 	now = time.strftime('%Y%m%d%H%M%S')
# 	print sub
# 	sub.pred[sub.pred<0] = 0
# 	sub.to_csv('res/'+now+'xgb.csv', header=False)


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
def aheadmutilpredict(step, ahead, tag, item):
	seq = mutilgetseq(tag)
	res = np.zeros(len(item))
	index = 0
	for pid in item.id:
		s = mutilgetith(seq, pid)
		flag,ts = mutilclean(s)
		w = [1,1]
		pred = 0
		if flag == 0:
			# pdb.set_trace()
			f,pre_x = mutilgemfeature(ts, step, ahead)
			y = f[:, 0]
			x = f[:, 1:f.shape[1]]
			pre = xgbtrain(x, y, pre_x)
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
		# break
		print pid,pred
	dat = pd.DataFrame([tag]*len(item), index = item.id, columns = ['type'])
	dat['pred'] = res
	return dat

def mutilsubmit():
	step = 14
	ahead = 14
	item = getitem()
	sub = aheadmutilpredict(step,ahead,'all', item)
	for i in range(1,6):
		d = aheadmutilpredict(step,ahead,str(i), item)
		sub = sub.append(d)
	now = time.strftime('%Y%m%d%H%M%S')
	print sub
	sub.pred[sub.pred<0] = 0
	sub.to_csv('res/'+now+'xgb_mutil_'+step+'_'+ahead+'.csv', header=False)
	# print sub


#nnet
# https://github.com/hawk31/nnet-ts
# def nnettrain(x, step, ahead):
# 	pdb.set_trace()
# 	ts = np.array(x)
# 	neural_net = TimeSeriesNnet(hidden_layers = [20,15], activation_functions = ['sigmoid', 'sigmoid'])
# 	neural_net.fit(ts, lag = step, epochs = 10000)
# 	neural_net.predict_ahead(n_ahead = ahead)
# 	res = neural_net.predictions
# 	return res


# def nnetpredict(step, ahead, tag):
# 	seq = getseq(tag)
# 	res = np.zeros(seq.shape[0])
# 	index = 0
# 	for s in seq.index:
# 		product = s
# 		flag,ts = clean(seq.ix[s])
# 		pred = 0
# 		if flag == 0:
# 			pdb.set_trace()
# 			pre = nnettrain(ts,step, ahead)
# 			pred = sum(pre)
# 		else:
# 			a = len(ts[ts>0])
# 			if a == 0:
# 				pred = 0;
# 			else:
# 				pred = sum(ts)/a*ahead;
# 		res[index] = pred
# 		index += 1
# 		break
# 		print product,pred
# 	dat = pd.DataFrame([tag]*seq.shape[0], index = seq.index, columns = ['type'])
# 	dat['pred'] = res
# 	return dat

# def nnetsubmit():
# 	step = 30
# 	ahead = 14
# 	sub = nnetpredict(step,ahead,'all')
# 	for i in range(1,6):
# 		d = nnetpredict(step,ahead,str(i))
# 		sub = sub.append(d)
# 	now = time.strftime('%Y%m%d%H%M%S')
# 	print sub
# 	sub.pred[sub.pred<0] = 0
# 	sub.to_csv('res/'+now+'nnet_7_14.csv', header=False)
# 	# print sub


if __name__ == '__main__':
	# pred = seqpredict(14,14)
	# submit()
	mutilsubmit()
	# nnetsubmit()