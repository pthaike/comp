#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
	# pred = seqpredict(14,14)
	df = pd.read_csv('submit0420.csv',header = None, names= ['id', 'type', 'pred'])
	# print df.ix(1)
	print df
	print df[df.pred<0]
	# df.pred[df.pred<0] = 0
	# df.to_csv('submit0420.csv',header=False, index = False)