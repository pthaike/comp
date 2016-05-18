#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
import pdb
from sklearn import grid_search
import xgboost as xgb
from musicpro import *
import time
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

_submit = False
# _submit = True
testnum = 30
_step = 28
_ahead = 60


def xgbpredict(x, y, pre_x):
	x, pre_x = datscater(x, pre_x)
	xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2, random_state=0)

	dtrain = xgb.DMatrix(xtrain, label = ytrain, missing = -1)
	dvalid = xgb.DMatrix(xvalid, label = yvalid, missing = -1)
	dpre = xgb.DMatrix(pre_x)

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
	model = xgb.train(param, dtrain, num_boost_round=1000, evals=watchlist)

	# model.save_model('xgb.model')
	print 'predict....'
	#predict
	pre_y = model.predict(dpre, ntree_limit=model.best_iteration)
	# printscore(ytest,pre_y)
	return pre_y

def score(trust, pre):
	e = trust - pre
	summ = 0
	for i in range(len(trust)):
		s = (e[i]/trust[i])**2
		summ = summ + s
	tho = np.sqrt(summ/len(trust))
	f = (1-tho)*np.sqrt(sum(trust))
	return f

def datscater(x, pre_x):
	scaler = StandardScaler()
	scaler.fit(x)
	X_train = scaler.transform(x)
	X_test = scaler.transform(pre_x)
	return X_train, X_test


def gbdrtrain(x, y, pre_x):
	x, pre_x = datscater(x, pre_x)
	clf = GradientBoostingRegressor(n_estimators=300,max_leaf_nodes =20, learning_rate=0.1,max_depth=6, random_state=400, loss='ls').fit(x, y)
	pred = clf.predict(pre_x)
	return pred

def svntrain(x, y, pre_x):
	x, pre_x = datscater(x, pre_x)
	clf = SVR().fit(x, y)
	pred = clf.predict(pre_x)
	return pred

def bayesiantrain(x, y, pre_x):
	x, pre_x = datscater(x, pre_x)
	clf = linear_model.BayesianRidge().fit(x, y)
	pred = clf.predict(pre_x)
	return pred

def LassoLarstrain(x, y, pre_x):
	x, pre_x = datscater(x, pre_x)
	clf = linear_model.LassoLars().fit(x, y)
	pred = clf.predict(pre_x)
	return pred

def nusvrtrain(x, y, pre_x):
	x, pre_x = datscater(x, pre_x)
	clf = NuSVR(C = 5.0).fit(x, y)
	pred = clf.predict(pre_x)
	return pred

def rfrtrain(x, y, pre_x):
	x, pre_x = datscater(x, pre_x)
	clf = RandomForestRegressor(n_estimators=300,max_leaf_nodes =20, max_depth=6, random_state=400).fit(x, y)
	pred = clf.predict(pre_x)
	return pred

def voting(xtrain, y, x_pre):
	ytrain = np.ravel(y)
	pre = 0.2*xgbpredict(xtrain, ytrain, x_pre) + 0.2*svntrain(xtrain, ytrain, x_pre) + 0.1*gbdrtrain(xtrain, ytrain, x_pre) + 0.2*nusvrtrain(xtrain, ytrain, x_pre) + 0.2*rfrtrain(xtrain, ytrain, x_pre) + 0.05*LassoLarstrain(xtrain, ytrain, x_pre) + 0.05*bayesiantrain(xtrain, ytrain, x_pre)
	return pre
	


def predict(ts, collect, down, step):
	if not _submit:
		aheadnum = testnum
	else:
		aheadnum = _ahead
	prediction = np.zeros(aheadnum)
	for i in range(aheadnum):
		# x, y, x_pre = genfeature(ts, step+i)
		x, y, x_pre = genmutilfeature(ts, down, collect, step+i, step)
		# pdb.set_trace()
		m, n = x.shape
		if not _submit:
			xtrain = x[0 : m - testnum]
			ytrain = y[0 : m - testnum]
			x_pre = x[m - testnum]
		else:
			xtrain = x
			ytrain = y

		x_pre = np.array([x_pre])

		# xgboost
		# pre = xgbpredict(xtrain, ytrain, x_pre)

		#gbdr
		# ytrain = np.ravel(ytrain)
		# pre = gbdrtrain(xtrain, ytrain, x_pre)
		# print "gbdr"

		#svn
		ytrain = np.ravel(ytrain)
		pre = svntrain(xtrain, ytrain, x_pre)
		print "svn"

		#nusvr
		# ytrain = np.ravel(ytrain)
		# pre = nusvrtrain(xtrain, ytrain, x_pre)
		# print "nusvr"

		#randomforest
		# ytrain = np.ravel(ytrain)
		# pre = rfrtrain(xtrain, ytrain, x_pre)
		# print "rfr"
		
		# voting
		# pre = voting(xtrain, ytrain, x_pre)
		# print "voting"

		prediction[i] = pre
		ts = np.concatenate((ts,pre), axis = 0)
	if not _submit:
		yt = y[m - testnum: m].T[0]
		f = score(yt, prediction)
		return f
	else:
		return prediction


#only the play feature
def predict_back(ts, step):
	if not _submit:
		aheadnum = testnum
	else:
		aheadnum = _ahead
	prediction = np.zeros(aheadnum)
	for i in range(aheadnum):
		x, y, x_pre = genfeature(ts, step+i)
		pdb.set_trace()
		m, n = x.shape
		if not _submit:
			xtrain = x[0 : m - testnum]
			ytrain = y[0 : m - testnum]
			x_pre = x[m - testnum]
		else:
			xtrain = x
			ytrain = y

		x_pre = np.array([x_pre])

		# xgboost
		# pre = xgbpredict(xtrain, ytrain, x_pre)

		#gbdr
		ytrain = np.ravel(ytrain)
		pre = gbdrtrain(xtrain, ytrain, x_pre)

		prediction[i] = pre
		ts = np.concatenate((ts,pre), axis = 0)
	if not _submit:
		yt = y[m - testnum: m].T[0]
		f = score(yt, prediction)
		return f
	else:
		return prediction

def submit():
	art = getartist()
	daterange = pd.period_range('20150901', '20151030', freq='D')
	date = [d.strftime('%Y%m%d') for d in daterange]
	subresult = pd.DataFrame()
	F = 0
	count = 0
	for aid in art.id:
		print "===============================================================>",count/float(len(art.id))
		d = getdat(aid)
		pre = predict(d[:, 0], d[:, 1], d[:, 2], _step)
		# pdb.set_trace()
		if not _submit:
			F += pre
		else:
			dat = pd.DataFrame([aid]*_ahead, columns = ['id'])
			dat['pred'] = pre
			dat['time'] = date
			subresult = subresult.append(dat)
		count += 1
		# if count == 3:
		# 	break
		print F
	print F
	if _submit:
		now = time.strftime('%Y%m%d%H%M%S')
		subresult.pred = np.round(subresult.pred).astype(int)
		subresult.to_csv('res/rfr'+now+'.csv', header = False, index = False)

if __name__ == '__main__':
	submit()