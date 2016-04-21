#combinning: all *.fits data
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import glob

def loaddata():
	path1='/home/vivek/0021_superEarths/3days/'
	step = 15
	index = 0
	name0 = []
	name1 = []
	transits = []
	initial_location = []
	period = 3
	star = dict()

#locating first transit
	for filename in glob.glob(os.path.join(path1,'*q01.fits')):
		name0.append(filename)
	print(len(name0))
	for i in range(len(name0)):
		data0 = fits.getdata(name0[i])
		time = data0[0]
		initial_location.append(str(time[0]) + str(period))
	print(initial_location)

#length estimation
	for filename in glob.glob(os.path.join(path1,'*.fits')):
		name1.append(filename)
	print(len(name1))
	summ = 0
	for i in range(len(name1)):
		data1 = fits.getdata(name1[i])
		time = data1[0]
		length = len(time)
		summ = summ + int(length/step)
	print (summ)

#loading data into matrix
	for filename in glob.glob(os.path.join(path1, '*.fits')):
		dir_len = len(path1)
		name_tmp = filename[dir_len:].split('_')
		star_name = name_tmp[0]
		num_tmp = int(name_tmp[1][1:-5])
		if star_name in star:
			star[star_name] = max(star[star_name], num_tmp)
		else:
			star[star_name] = num_tmp
	s = (15, 257862)
	trainx = np.zeros(s)
	trainy = np.zeros(257862)
	il = 0
	for key in star:
		time = []
		flux = []
		max_num = star[key]
		for i in xrange(1,max_num):
			if i>=10:		
				file_name = key+"_q"+str(i)+'.fits'
			else:
				file_name = key+"_q0"+str(i)+'.fits'
			data2 = fits.getdata('/home/vivek/0021_superEarths/3days/'+file_name)
			time.extend(data2[0])
			flux.extend(data2[1])
		initialtime = initial_location[il]
		length = len(time)
		print length
		k = 0
		t = 0
		while(k < time[length-1]):
			k = float(initialtime) + t*period
			transits.append(k)
			t = t + 1
		print len(transits)
		initial = 0
		for j in range(int(length/step)):
			fluxtem = flux[initial:initial+step]
			timetem = time[initial:initial+step]
			trainy[j+index] = 0
			plt.plot(timetem, fluxtem)
			for m in range(len(transits)):
				if(time[initial] < transits[m] and transits[m] < time[initial + step]):
					trainy[j+index] = 1
			print(trainy[j+index])
			plt.show()
			trainx[:, j+index] = fluxtem
			initial = initial + step
		index = index + int(length/step)
		il = il + 1
	p = 'x_0021superearth_3days.txt'
	q = 'y_0021superearth_3days.txt'
	np.savetxt(p, trainx, delimiter=',')
	np.savetxt(q, trainy, delimiter=',')
'''
#main function
	s=(100,31982)
	trainx=np.zeros(s)
	trainy=np.zeros()
	for i in range(len(name1)):
		data = fits.getdata(name1[i])
		print (name1[i], i, i/len(name1))
		time=data[0]
		flux=data[1]
		initialtime = time[0]
		length=len(time)
		k = 0
		t = 1
		print (time[length-1])
		while(k<time[length-1]): #locating transits in time
			k=initialtime+t*period
			transits.append(k)
			t=t+1
		initial = 0
		for j in range(0, int(length/step)):
			fluxtemp=flux[initial:initial+step]
			timetemp=time[initial:initial+step]
			trainy[j+index] = 0
			plt.plot(timetemp,fluxtemp)
			for m in range(len(transits)):
				if(time[initial]<transits[m] and transits[m]<time[initial+step]):
                    			trainy[j+index] = 0
			print(trainy[j+index])
			plt.show()
			#char = input("please stop")
			trainx[:,j+index]=fluxtemp
			initial = initial+step
		index = index + int(length/step)
		#print (index)
	p = 'x_0021_800_0331.csv'
	q = 'y_0021_800_0331.csv'
	np.savetxt(p,trainx,delimiter=",")
	np.savetxt(q,trainy,delimiter=",")
'''

loaddata()
