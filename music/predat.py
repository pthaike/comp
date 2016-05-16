#! /usr/bin/python
# encoding: utf-8

#获取歌曲艺人id
def getartlist(file):
	list = []
	num = 0
	fr = open(file,'r')
	for line in fr.readlines():
		line = line.strip()
		li = line.split(',')
		if(li[1] not in list):
			list.append(li[1])
			num += 1
	fr.close()
	return list

#获取歌曲id
def getsonglist(file):
	list = []
	num = 0
	fr = open(file,'r')
	for line in fr.readlines():
		line = line.strip()
		li = line.split(',')
		if(li[0] not in list):
			list.append(li[0])
			num += 1
	fr.close()
	return list

#获取歌曲id
def getsonglist(file):
	list = []
	num = 0
	fr = open(file,'r')
	for line in fr.readlines():
		line = line.strip()
		li = line.split(',')
		if(li[0] not in list):
			list.append(li[0])
			num += 1
	fr.close()
	return list


# main
if __name__ == '__main__':
	fw = open('song.csv','w')
	list = getsonglist('mars_tianchi_songs.csv')
	index = 0
	for d in list:
		fw.write(d)
		fw.write(',')
		fw.write(str(index))
		fw.write('\n')
		index += 1
	fw.close()
