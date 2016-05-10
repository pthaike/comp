#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
	f1 = '20160509175957xgb_mutil_have28_14_14.csv'
	f2 = '20160509192835xgb_mutil_have28_14_14.csv'
	df1 = pd.read_csv(f1,header = None, names= ['id', 'type', 'pred'])
	df2 = pd.read_csv(f2,header = None, names= ['id', 'type', 'pred'])
	df1.pred = df1.pred * 0.8+ df2.pred * 0.2

	df1.to_csv('050917_050919_submit.csv',header=False, index = False)