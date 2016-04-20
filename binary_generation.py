import numpy as np
from random import randint
from numpy import newaxis
import random

# Generating periodical data for training

a = np.random.randint(low=10, high=400, size=100)
print(len(a))
print(len(np.unique(a)))
sequence_0 = np.zeros(800)
init_point0 = int(random.random()*a[0])
for init_point0 in range(800):
	sequence_0[init_point0] = sequence_0[init_point0] + 1
	init_point0 = init_point0 + a[0]
sequence_0 = sequence_0[newaxis, :]
for x in range(0, 299):
	sequence_start = np.zeros(800)
	y = int(random.random()*a[0])
	for y in range(800):
		sequence_start[y] = sequence_start[y] + 1
		y = y + a[0]
	sequence_0 = np.concatenate((sequence_0, sequence_start[newaxis, :]), axis=0)
print(sequence_0.shape)
j = 1
for j in range(1, 100):
    for k in range(300):
        sequence_init = np.zeros(800)
        i = int(random.random()*a[j])
        while(i < 800):
            sequence_init[i] = sequence_init[i] + 1
            i = i + a[j]
        sequence_init = sequence_init[newaxis, :]
        sequence_0 = np.concatenate((sequence_0, sequence_init), axis=0)
print(sequence_0.shape)
label_p = np.zeros(30000)
label_p = label_p + 1
label_p = label_p[:, newaxis]
training_p = np.concatenate((sequence_0, label_p), axis=1)
print(training_p.shape)
np.savetxt('training_p.txt', training_p, delimiter=',', header='')

#Generating while noise

noise = np.zeros(3200000)
i = randint(1, 300)
for i in range(0, 3200000):
        noise[i] = randint(0, 1)
#print(noise)
print(noise.shape)
noise = np.reshape(noise, (4000, 800))
print(noise.shape)
label_n= np.zeros(4000)
label_n = label_n[:, newaxis]
training_n = np.concatenate((noise, label_n), axis=1)
print(training_n.shape)
#x = input('...')
np.savetxt('training_n.txt', training_n, delimiter=',')

#Generating slightly different 'non-periodical' data
b = np.random.randint(low=10, high=400, size=100)
print(len(b))
print(len(np.unique(b)))
sequence_1 = np.zeros(800)
init_point1 = int(( 1 + random.random())*b[0])
for init_point1 in range(800):
    sequence_1[init_point1] = sequence_1[init_point1] + 1
    init_point1 = init_point1 + b[0]*int(random.random())
sequence_1 = sequence_1[newaxis, :]
for x in range(0, 299):
    sequence_start = np.zeros(800)
    y = int(random.random()*b[0])
    for y in range(800):
        sequence_start[y] = sequence_start[y] + 1
        y = y + int(b[0]*(random.random()))
    sequence_start = sequence_start[newaxis, :]
    sequence_1 = np.concatenate((sequence_1, sequence_start), axis=0)
print(sequence_1.shape)
j = 1
for j in range(1, 100):
    k = 0
    for k in range(300):
        sequence_init = np.zeros(800)
        i = int(random.random()*b[j])
        while(i < 800):
            sequence_init[i] = sequence_init[i] + 1
            i = i + int(b[j]*random.random())
        sequence_init = sequence_init[newaxis, :]
        sequence_1 = np.concatenate((sequence_1, sequence_init), axis=0)
print(sequence_1.shape)
label_s = np.zeros(30000)
label_s = label_s[:, newaxis]
training_s = np.concatenate((sequence_1, label_s), axis=1)
print(training_s.shape)
np.savetxt('training_s.txt', training_p, delimiter=',', header='')

temp_matrix = np.concatenate((training_p, training_n), axis=0)
matrix = np.concatenate((temp_matrix, training_s), axis=0)
matrix = np.random.permutation(matrix)
print(matrix.shape)
np.savetxt('training.txt', matrix, delimiter=',')

#Generating periodical data for testing with 1 label

c = np.random.randint(low=10, high=500, size=90)
print(len(c))
print(len(np.unique(c)))
a = a[newaxis, :]
c = c[newaxis, :]
ac = np.concatenate((a, c), axis=1)
print(len(np.unique(ac)))
sequence_2 = np.zeros(800)
init_point2 = int((random.random())*c[0])
for init_point2 in range(800):
    sequence_2[init_point2] = sequence_2[init_point2] + 1
    init_point2 = init_point2 + c[0]
sequence_2 = sequence_2[newaxis, :]
for x in range(0, 99):
    sequence_start = np.zeros(800)
    y = int(random.random()*c[0])
    for y in range(800):
        sequence_start[y] = sequence_start[y] + 1
        y = y + c[0]
    sequence_start = sequence_start[newaxis, :]
    sequence_2 = np.concatenate((sequence_2, sequence_start), axis=0)
print(sequence_2.shape)
j = 1
for j in range(1, 90):
    k = 0
    for k in range(90):
        sequence_init = np.zeros(800)
        i = int(random.random()*c[j])
        while(i < 800):
            sequence_init[i] = sequence_init[i] + 1
            i = i + c[j]
        sequence_init = sequence_init[newaxis, :]
        sequence_2 = np.concatenate((sequence_2, sequence_init), axis=0)
print(sequence_2.shape)
label_p1 = np.zeros(9000)
label_p1 = label_p1 + 1
label_p1 = label_p1[:, newaxis]
testing_p1 = np.concatenate((sequence_2, label_p1), axis=1)
print(testing_p1.shape)
np.savetxt('testing_p1.txt', testing_p1, delimiter=',', header='')

#Generating slightly different periodical data for testing label=0

d = np.random.randint(low=10, high=500, size=80)
print(len(d))
print(len(np.unique(d)))
b = b[newaxis, :]
d = d[newaxis, :]
bd = np.concatenate((b, d), axis=0)
print(len(np.unique(bd)))
sequence_3 = np.zeros(800)
init_point3 = int(random.random()*d[0])
for init_point3 in range(800):
    sequence_3[init_point3] = sequence_3[init_point3] + 1
    init_point3 = init_point3 + b[0]*int(random.random())
sequence_3 = sequence_3[newaxis, :]
for x in range(0, 99):
    sequence_start = np.zeros(800)
    y = int(random.random()*d[0])
    for y in range(800):
        sequence_start[y] = sequence_start[y] + 1
        y = y + int(d[0]*(random.random()))
    sequence_start = sequence_start[newaxis, :]
    sequence_3 = np.concatenate((sequence_3, sequence_start), axis=0)
print(sequence_3.shape)
j = 1
for j in range(1, 80):
    k = 0
    for k in range(80):
        sequence_init = np.zeros(800)
        i = int(random.random()*d[j])
        while(i < 800):
            sequence_init[i] = sequence_init[i] + 1
            i = i + int(d[j]*random.random())
        sequence_init = sequence_init[newaxis, :]
        sequence_3 = np.concatenate((sequence_3, sequence_init), axis=0)
print(sequence_3.shape)
label_p0 = np.zeros(8000)
label_p0= label_p0
label_p0 = label_p0[:, newaxis]
testing_p0 = np.concatenate((sequence_3, label_p0), axis=1)
print(testing_p0.shape)

testing_temp = np.concatenate((matrix[50000:64000, :], testing_p1), axis=0)
testing = np.concatenate((testing_temp, testing_p0), axis=0)
testing = np.random.permutation(testing)
print(testing.shape)
print(testing)
np.savetxt('testing.txt', testing, delimiter=',')
