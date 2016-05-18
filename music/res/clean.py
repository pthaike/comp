#! /usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd

# _submit = True

if __name__ == '__main__':
	df = pd.read_csv('gbrt20160516210356.csv', header = None, names = ['id', 'predict', 'time'])
	df.predict = np.round(df.predict).astype(int)
	df.to_csv('submit0518_xgb_14.csv', header = False, index = False)