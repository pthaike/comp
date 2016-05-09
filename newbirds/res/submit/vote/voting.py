#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
	f1 = 'submit_20160422105437xgb_mutil_14_14.csv'
	f2 = 'submit_20160425201255xgb_mutil_avg_14_14.csv'
	df1 = pd.read_csv(f1,header = None, names= ['id', 'type', 'pred'])
	df2 = pd.read_csv(f2,header = None, names= ['id', 'type', 'pred'])
	df1.pred = df1.pred * 0.5 + df2.pred * 0.5

	df1.to_csv('submit.csv',header=False, index = False)