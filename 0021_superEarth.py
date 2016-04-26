#combinning: all *.fits data
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import glob

def loaddata():
    path1='/home/vivek/anaconda2/0021_superEarths/0021_superEarths/3days/'
    step = 1000
    transits = []
    initial_location = []
    period = 3
    star = dict()
    
    for filename in glob.glob(os.path.join(path1, '*.fits')):
        dir_len = len(path1)
        name_tmp = filename[dir_len:].split('_')
        star_name = name_tmp[0] #this is the star's name
        num_tmp = int(name_tmp[1][1:3]) #this is quarter number given in the filename
        if star_name in star:
            star[star_name] = max(star[star_name], num_tmp)
        else:
            star[star_name] = num_tmp
        '''through this for loop, for every star, we get the maximum quarter number 
        it has and hence populate the star dictionary'''

    s = (step, 121064)
    trainx = np.zeros(s)
    trainy = np.zeros(121064)
    il = 0
    index = 0

    for key in star:
        time = []
        flux = []
        max_num = star[key]
        for i in xrange(1,max_num+1):
            if i>=10:
                file_name = key+"_q"+str(i)+'.fits'
            else:
                file_name = key+"_q0"+str(i)+'.fits'
            data2 = fits.getdata('/home/vivek/0021_superEarths/0021_superEarths/3days/'+file_name)
            time.extend(data2[0])
            flux.extend(data2[1])
        initialtime = time[0] + period
        
        '''through this for loop, we get the time and flux matrix for all the stars 
        spanning the max/corresponding number of quarters for which a star recorded activity. So,
        if star 1 has activity for 14 quarters, this for loop will run 14 times to collect all the
        time and flux values.'''
        
        k = 0
        t = 0
        
        while(k < time[-1]):
            k = float(initialtime) + t*period
            transits.append(k)
            t = t + 1
        #print len(transits)
        initial = 0
        
        for j in range(int(math.floor(len(time)/step)-1)):
            fluxtem = flux[initial:initial+step]
            timetem = time[initial:initial+step]
            trainy[j+index] = 0
            plt.plot(timetem, fluxtem)
            plt.show
            for m in range(len(transits)):
                if(time[initial] < transits[m] and transits[m] < time[initial + step]):
                    trainy[j+index] = 1
            #print(trainy[j+index])
            #plt.show()
            trainx[:, j+index] = fluxtem
            initial = initial + step
        
        index = index + int(math.floor(len(time)/step)-1)
    
    p = 'x_0021superearth_3days.txt'
    q = 'y_0021superearth_3days.txt'
    np.savetxt(p, trainx, delimiter=',')
    np.savetxt(q, trainy, delimiter=',')

loaddata()
