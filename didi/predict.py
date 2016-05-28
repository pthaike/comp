#! /usr/bin/python

import pandas as pd 
import numpy as np 
from process import *
from datetime import timedelta
import time
import pdb

_distric = 66
_submit = True
	

def submit():
	date = readpredict()
	subresult = pd.DataFrame()
	enddate = pd.to_datetime('2016-01-21')
	for d in date.date:
		l = d.strip().split('-')
		t = l[0]+'-'+l[1]+'-'+l[2]
		timeslice = int(l[3].strip())
		t = pd.to_datetime(t)
		t = t + timedelta(days = -7)
		# pdb.set_trace()
		if t > enddate:
			t = t + timedelta(days = -7)
		t2 = t + timedelta(days = -7)
		prediction = np.zeros(_distric)
		print t
		print t2
		for distric in range(1, _distric + 1):
			
			f1 = t.strftime('%Y-%m-%d') + '-' + str(distric)
			df1 = readdat(f1)
			pred1 = (df1.no[timeslice-1] + df1.no[timeslice] + df1.no[timeslice-2])/3.0
			
			f2 = t2.strftime('%Y-%m-%d') + '-' + str(distric)
			df2 = readdat(f2)
			pred2 = (df2.no[timeslice-1] + df2.no[timeslice] + df2.no[timeslice-2])/3.0
			prediction[distric-1] = (pred1 + pred2)/2.0
		dat = pd.DataFrame(prediction, columns = ['pred'])
		dat['id'] = range(1, _distric + 1)
		dat['md'] = [d] * _distric
		subresult = subresult.append(dat)
	if _submit:
		now = time.strftime('%Y%m%d%H%M%S')
		subresult.id = np.round(subresult.id).astype(int)
		subresult.to_csv('res/'+now+'.csv', header = False, index = False)


if __name__ == '__main__':
	submit()