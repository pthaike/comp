#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
from musicpro import *
import time


_submit = False
# _submit = True
testnum = 60
_step = 14
_ahead = 60


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


def voting(xtrain, y, x_pre):
	ytrain = np.ravel(y)
	pre = 0.3*xgbpredict(xtrain, ytrain, x_pre) + 0.3*rfrtrain(xtrain, ytrain, x_pre) + 0.1*LassoLarstrain(xtrain, ytrain, x_pre) + 0.1*bayesiantrain(xtrain, ytrain, x_pre) + 0.1*gbdrtrain(xtrain, ytrain, x_pre) + 0.1*nusvrtrain(xtrain, ytrain, x_pre)
	return pre


def predict(ts, step):
	w = []
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
		# s1 = 0
		# for k in range(1,step+1):
		# 	s1 += np.mean(ts[len(ts) - k: len(ts)])
		# s1 = s1 / float(step)
		# s2 = np.mean(ts[len(ts) - step: len(ts)])
		# wind =  2*s1 - s2 + 2*(s1-s2)/(step-1)
		# pre = wind

		pre = 0.7 * ts[-1] + 0.3 * np.mean(ts[len(ts) - step: len(ts)])

		pre = np.array([pre])
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
		topk = gettopk(aid)
		pre = predict(d[:, 0],_step)
		# pdb.set_trace()
		
		if not _submit:
			F += pre
		else:
			if aid == '2b7fedeea967becd9408b896de8ff903':
				pre = np.ones(pre.shape) 
			dat = pd.DataFrame([aid]*_ahead, columns = ['id'])
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
		subresult.to_csv('res/voting'+now+'.csv', header = False, index = False)

if __name__ == '__main__':
	submit()
	print _step
	# print "xgb"
	# print "pre = 0.3*xgbpredict(xtrain, ytrain, x_pre) + 0.3*rfrtrain(xtrain, ytrain, x_pre) + 0.1*LassoLarstrain(xtrain, ytrain, x_pre) + 0.1*bayesiantrain(xtrain, ytrain, x_pre) + 0.1*gbdrtrain(xtrain, ytrain, x_pre) + 0.1*nusvrtrain(xtrain, ytrain, x_pre)"