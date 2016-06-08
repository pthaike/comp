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
# from libnnet import *
from sklearn.feature_selection import SelectFromModel

# _submit = False
_submit = True
testnum = 60
_step = 14
_ahead = 61
_time = 30


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
		'silent' : 1,
		'colssample_bytree':0.8,
		'eta':0.02,
		'nthread':10,
		'seed':400
	}

	watchlist = [ (dtrain,'train'), (dvalid,'val')]
	model = xgb.train(param, dtrain, num_boost_round=500, evals=watchlist)

	# model.save_model('xgb.model')
	print 'predict....'
	#predict
	pre_y = model.predict(dpre, ntree_limit=model.best_iteration)
	# printscore(ytest,pre_y)
	return pre_y

def score(trust, pre):
	e = trust - pre
	# pdb.set_trace()
	summ = 0
	for i in range(trust.shape[0]):
		# pdb.set_trace()
		if trust[i] == 0:
			continue
		s = (e[i]/trust[i])**2
		summ = summ + s
	# pdb.set_trace()
	tho = np.sqrt(summ/len(trust))
	f = (1-tho)*np.sqrt(sum(trust))
	if f < 0:
		# pdb.set_trace()
		print f
		print trust
		pre = np.ones(pre.shape)
		e = trust - pre
		summ = 0
		for i in range(trust.shape[0]):
			# pdb.set_trace()
			if trust[i] == 0:
				continue
			s = (e[i]/trust[i])**2
			summ = summ + s
		# pdb.set_trace()
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
	clf = GradientBoostingRegressor(n_estimators=740, min_samples_leaf = 0.8, min_samples_split = 40, learning_rate=0.1,max_depth=7, random_state=400, loss='huber').fit(x, y)
	# clf = GradientBoostingRegressor(n_estimators=200,max_leaf_nodes =20, learning_rate=0.1,max_depth=6, random_state=400, loss='ls').fit(x, y)

	pred = clf.predict(pre_x)
	return pred

def paramselect(x, y, pre_x):
	#740
	# param_test1 = {'n_estimators':range(600,800,5)}
	# gsearch1 = grid_search.GridSearchCV(estimator=
	# 	GradientBoostingRegressor(learning_rate=0.1,min_samples_split=20,
	# 		min_samples_leaf=50,max_depth=6,max_features='sqrt',
	# 		subsample=0.8,random_state=200),
	# 	param_grid=param_test1, scoring='mean_squared_error',iid=False,cv=5)

	#40,7
	# param_test2 = {'max_depth':range(7,12,1), 'min_samples_split':range(30,100,10)}
	# gsearch2 = grid_search.GridSearchCV(estimator = 
	# 	GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 740, max_features = 'sqrt', subsample=0.8, random_state=200),
	# 	param_grid=param_test2, scoring='mean_squared_error', iid=False, cv=5)
	# clf = gsearch2

	#test min_samples_leaf 30
	# param_test3 = {'min_samples_leaf':range(20,210,5)}
	# gsearch3 = grid_search.GridSearchCV(estimator = 
	# 	GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 740, max_depth=7, min_samples_split=40, max_features = 'sqrt', subsample=0.8, random_state=200),
	# 	param_grid=param_test3, scoring='mean_squared_error', iid=False, cv=5)
	# clf = gsearch3

	#0.8
	param_test5 = {'subsample':[0.6, 0.7, 0.8,0.9]}
	gsearch5 = grid_search.GridSearchCV(estimator = 
		GradientBoostingRegressor(learning_rate = 0.1, min_samples_leaf = 30, n_estimators = 740, max_depth=7, min_samples_split=40, random_state=200),
		param_grid=param_test5, scoring='mean_squared_error', iid=False, cv=5)
	clf = gsearch5

	clf = clf.fit(x, y)
	print clf.grid_scores_,'\n', clf.best_params_, clf.best_score_

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
	# clf = RandomForestRegressor(n_estimators=740,max_leaf_nodes =20, min_samples_leaf = 30, min_samples_split=40, max_depth=7, random_state=400, n_jobs = 6).fit(x, y)
	clf = RandomForestRegressor(n_estimators=200,max_leaf_nodes =20, max_depth=6, random_state=400, n_jobs = 6).fit(x, y)

	# clf = RandomForestRegressor(n_estimators=300,max_leaf_nodes =20, max_depth=6, random_state=400).fit(x, y)
	pred = clf.predict(pre_x)
	return pred

def voting(xtrain, y, x_pre):
	ytrain = np.ravel(y)
	pre = 0.3*xgbpredict(xtrain, ytrain, x_pre) + 0.3*rfrtrain(xtrain, ytrain, x_pre) + 0.1*LassoLarstrain(xtrain, ytrain, x_pre) + 0.1*bayesiantrain(xtrain, ytrain, x_pre) + 0.1*gbdrtrain(xtrain, ytrain, x_pre) + 0.1*nusvrtrain(xtrain, ytrain, x_pre)
	return pre

def extratrees(x, y, pre_x):
	x, pre_x = datscater(x, pre_x)
	clf = RandomForestRegressor(n_estimators=200,max_leaf_nodes =20, max_depth=6, random_state=400).fit(x, y)
	# clf = RandomForestRegressor(n_estimators=300,max_leaf_nodes =20, max_depth=6, random_state=400).fit(x, y)
	pred = clf.predict(pre_x)
	print clf.feature_importances_
	return pred

def selectfeature(x, y, x_pre):
	x, x_pre = datscater(x, x_pre)
	clf = linear_model.LassoLars().fit(x, y)
	model = SelectFromModel(clf, prefit=True)
	x_new = model.transform(x)
	print 'x',x.shape
	print x_new.shape
	x_pre = model.transform(x_pre)
	return x_new, x_pre

def predict(ts, collect, down, topk, step, flag):
	# print "randomforest"
	print 'flag',flag
	if not _submit:
		aheadnum = testnum
	else:
		aheadnum = _ahead
	# pdb.set_trace()
	yt = ts[ts.shape[0] - testnum: ts.shape[0]]
	prediction = np.zeros(aheadnum)
	for i in range(aheadnum):
		# x, y, x_pre = genmutilfeaturemoretopk(ts, down, collect, topk, step+i, step)
		x, y, x_pre = genmutilfeaturemore(ts, down, collect, step+i, step)
		# x, y, x_pre = genmutilfeature(ts, down, collect, step+i, step)
		# pdb.set_trace()
		m, n = x.shape
		if not _submit:
			xtrain = x[m - testnum - _time : m - testnum]
			ytrain = y[m - testnum - _time : m - testnum]
			x_pre = x[m - testnum]
		else:
			xtrain = x[m - _time:]
			ytrain = y[m - _time:]

		x_pre = np.array([x_pre])

		# xtrain, x_pre = selectfeature(xtrain, ytrain, x_pre)

		# xgboost
		# pre = xgbpredict(xtrain, ytrain, x_pre)
		# print xgb

		#gbdr
		# ytrain = np.ravel(ytrain)
		# pre = gbdrtrain(xtrain, ytrain, x_pre)
		# print "gbdr huber"

		#svn
		# ytrain = np.ravel(ytrain)
		# pre = svntrain(xtrain, ytrain, x_pre)
		# print "svn"

		#nusvr
		# ytrain = np.ravel(ytrain)
		# pre = nusvrtrain(xtrain, ytrain, x_pre)
		# print "nusvr"

		#randomforest
		# ytrain = np.ravel(ytrain)
		# pre = rfrtrain(xtrain, ytrain, x_pre)
		# print "rfr"

		#bayesiantrain
		# ytrain = np.ravel(ytrain)
		# pre = bayesiantrain(xtrain, ytrain, x_pre)

		#LassoLarstrain
		# ytrain = np.ravel(ytrain)
		# pre = LassoLarstrain(xtrain, ytrain, x_pre)
		
		# voting
		# pre = voting(xtrain, ytrain, x_pre)
		# print "voting"

		#extratrees
		# ytrain = np.ravel(ytrain)
		# pre = extratrees(xtrain, ytrain, x_pre)
		# print "extratrees"


		# ytrain = np.ravel(ytrain)
		# paramselect(xtrain, ytrain, x_pre)
		# pre = 0

		ytrain = np.ravel(ytrain)
		
		if flag == 1:
			pre = rfrtrain(xtrain, ytrain, x_pre)
		elif flag == 2:
			pre = gbdrtrain(xtrain, ytrain, x_pre)
		elif flag == 3:
			pre = svntrain(xtrain, ytrain, x_pre)
		elif flag == 4:
			pre = LassoLarstrain(xtrain, ytrain, x_pre)
		else:
			pre = rfrtrain(xtrain, ytrain, x_pre)



		prediction[i] = pre
		ts = np.concatenate((ts,pre), axis = 0)
	if not _submit:
		# yt = y[m - testnum: m].T[0]
		f = score(yt, prediction)
		return f
	else:
		return prediction[1:]

##20160531
def predict_back3(ts, collect, down, step):
	# print "randomforest"
	if not _submit:
		aheadnum = testnum
	else:
		aheadnum = _ahead
	# pdb.set_trace()
	yt = ts[ts.shape[0] - testnum: ts.shape[0]]
	prediction = np.zeros(aheadnum)
	for i in range(aheadnum):
		# x, y, x_pre = genmutilfeaturemoretopk(ts, down, collect, topk, step+i, step)
		x, y, x_pre = genmutilfeaturemore(ts, down, collect, step+i, step)
		# x, y, x_pre = genmutilfeature(ts, down, collect, step+i, step)
		# pdb.set_trace()
		m, n = x.shape
		if not _submit:
			xtrain = x[m - testnum - _time : m - testnum]
			ytrain = y[m - testnum - _time : m - testnum]
			x_pre = x[m - testnum]
		else:
			xtrain = x[m - _time:]
			ytrain = y[m - _time:]

		x_pre = np.array([x_pre])

		# xtrain, x_pre = selectfeature(xtrain, ytrain, x_pre)

		# xgboost
		# pre1 = xgbpredict(xtrain, ytrain, x_pre)
		# print xgb

		#gbdr
		# ytrain = np.ravel(ytrain)
		# pre = gbdrtrain(xtrain, ytrain, x_pre)
		# print "gbdr huber"

		#svn
		# ytrain = np.ravel(ytrain)
		# pre = svntrain(xtrain, ytrain, x_pre)
		# print "svn"

		#nusvr
		# ytrain = np.ravel(ytrain)
		# pre = nusvrtrain(xtrain, ytrain, x_pre)
		# print "nusvr"

		#randomforest
		ytrain = np.ravel(ytrain)
		pre = rfrtrain(xtrain, ytrain, x_pre)

		# pre = 0.4 * pre1 + 0.6 * pre2;
		# print "rfr"

		#bayesiantrain
		# ytrain = np.ravel(ytrain)
		# pre = bayesiantrain(xtrain, ytrain, x_pre)

		#LassoLarstrain
		# ytrain = np.ravel(ytrain)
		# pre = LassoLarstrain(xtrain, ytrain, x_pre)
		
		# voting
		# pre = voting(xtrain, ytrain, x_pre)
		# print "voting"

		#extratrees
		# ytrain = np.ravel(ytrain)
		# pre = extratrees(xtrain, ytrain, x_pre)
		# print "extratrees"


		# ytrain = np.ravel(ytrain)
		# paramselect(xtrain, ytrain, x_pre)
		# pre = 0




		prediction[i] = pre
		ts = np.concatenate((ts,pre), axis = 0)
	if not _submit:
		# yt = y[m - testnum: m].T[0]
		f = score(yt, prediction)
		return f
	else:
		return prediction[1:]

def predict_back2(ts, collect, down, step):
	# print "randomforest,selectfeature"
	if not _submit:
		aheadnum = testnum
	else:
		aheadnum = _ahead
	# pdb.set_trace()
	yt = ts[ts.shape[0] - testnum: ts.shape[0]]
	prediction = np.zeros(aheadnum)
	for i in range(aheadnum):
		x, y, x_pre = genmutilfeaturemore(ts, down, collect, step+i, step)
		# x, y, x_pre = genmutilfeaturemore(ts, down, collect, step+i, step)
		# x, y, x_pre = genmutilfeature(ts, down, collect, step+i, step)
		# pdb.set_trace()
		m, n = x.shape
		if not _submit:
			xtrain = x[0 : m - testnum]
			ytrain = y[0 : m - testnum]
			x_pre = x[m - testnum]
		else:
			xtrain = x
			ytrain = y

		# x_pre = np.array([x_pre])

		# xtrain, x_pre = selectfeature(xtrain, ytrain, x_pre)

		# xgboost
		pre = xgbpredict(xtrain, ytrain, x_pre)
		# print xgb

		#gbdr
		# ytrain = np.ravel(ytrain)
		# pre = gbdrtrain(xtrain, ytrain, x_pre)
		# print "gbdr"

		#svn
		# ytrain = np.ravel(ytrain)
		# pre = svntrain(xtrain, ytrain, x_pre)
		# print "svn"

		#nusvr
		# ytrain = np.ravel(ytrain)
		# pre = nusvrtrain(xtrain, ytrain, x_pre)
		# print "nusvr"

		#randomforest
		# ytrain = np.ravel(ytrain)
		# pre = rfrtrain(xtrain, ytrain, x_pre)
		# print "rfr"

		#bayesiantrain
		# ytrain = np.ravel(ytrain)
		# pre = bayesiantrain(xtrain, ytrain, x_pre)

		#LassoLarstrain
		# ytrain = np.ravel(ytrain)
		# pre = LassoLarstrain(xtrain, ytrain, x_pre)
		
		# voting
		# pre = voting(xtrain, ytrain, x_pre)
		# print "voting"

		#extratrees
		# ytrain = np.ravel(ytrain)
		# pre = extratrees(xtrain, ytrain, x_pre)
		# print "extratrees"


		prediction[i] = pre
		ts = np.concatenate((ts,pre), axis = 0)
	if not _submit:
		# yt = y[m - testnum: m].T[0]
		f = score(yt, prediction)
		return f
	else:
		return prediction



def netpredict(ts, step):
	print "nnet"
	if not _submit:
		aheadnum = testnum
	else:
		aheadnum = _ahead
	yt = ts[ts.shape[0] - testnum: ts.shape[0]]
	m = ts.shape[0]
	lag = 40
	if not _submit:
		ts = ts[0 : m - testnum]
	prediction = np.zeros(aheadnum)
	for i in range(aheadnum):
		# pdb.set_trace()
		# nnet
		neural_net = TimeSeriesNnet(hidden_layers = [20,15, 5], activation_functions = ['sigmoid', 'relu', 'softmax'])
		neural_net.fit(ts, lag = lag+i, epochs = 1000)
		neural_net.predict_ahead(n_ahead = 1)
		pre = neural_net.predictions
		# pdb.set_trace()
		prediction[i] = pre
		ts = np.concatenate((ts,pre), axis = 0)
	print prediction
	print yt
	if not _submit:
		f = score(yt, prediction)
		return f
	else:
		return prediction


#only the play feature
def predict_back1(ts, step):
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
			xtrain = x[60 : m - testnum]
			ytrain = y[60 : m - testnum]
			x_pre = x[m - testnum]
		else:
			xtrain = x[120:]
			ytrain = y[120:]

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
	#flag = getw()
	count = 0
	for aid in art.id:
		print "===============================================================>",count/float(len(art.id))
		d = getdat(aid)
		#topk = gettopk(aid)
		# pre = predict(d[:, 0], d[:, 1], d[:, 2], topk, _step, flag.flag[count])
		# pre = predict(d[:, 0], d[:, 1], d[:, 2], topk, _step)
		pre = predict_back3(d[:, 0], d[:, 1], d[:, 2], _step)
		# pre = predict_back2(d[:, 0], d[:, 1], d[:, 2], topk, _step)
		# pre = predict_back2(d[:, 0], d[:, 1], d[:, 2], _step)
		# pre = netpredict(d[:, 0], _step)
		# pdb.set_trace()
		
		if not _submit:
			F += pre
		else:
			if aid == '2b7fedeea967becd9408b896de8ff903':
				pre = np.ones(pre.shape) 
			dat = pd.DataFrame([aid]*len(pre), columns = ['id'])
			dat['pred'] = pre
			dat['time'] = date
			subresult = subresult.append(dat)
		count += 1
		# if count == 3:
		# 	break
		print aid
		print pre
		print F
	print F
	if _submit:
		now = time.strftime('%Y%m%d%H%M%S')
		subresult.pred = np.round(subresult.pred).astype(int)
		subresult.to_csv('res/all'+now+'.csv', header = False, index = False)

if __name__ == '__main__':
	submit()
	print _step
	# print "xgb"
	# print "pre = 0.3*xgbpredict(xtrain, ytrain, x_pre) + 0.3*rfrtrain(xtrain, ytrain, x_pre) + 0.1*LassoLarstrain(xtrain, ytrain, x_pre) + 0.1*bayesiantrain(xtrain, ytrain, x_pre) + 0.1*gbdrtrain(xtrain, ytrain, x_pre) + 0.1*nusvrtrain(xtrain, ytrain, x_pre)"