import numpy as np
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def read_data(filename,particles, line, timesteps, hat, cut):
    f=open(filename,"r")
    arr=np.empty(((timesteps-cut,particles,line)))
    #print(arr[1][1])
    for i in range(timesteps):
        if(i < cut):
            for ii in range(hat):
                f.readline()
            for ii in range(particles):
                f.readline()
            #print(i)
        else:
            for ii in range(hat):
                f.readline()
            for ii in range(particles):
                arr[i-cut][ii]=f.readline().split(' ')
            #print(i)
    return arr

def load_data(filename, particles, line):
    arr = np.loadtxt(filename)

    #print(arr.shape)

    arr = arr.reshape(arr.shape[0], particles, line)

    #print(arr.shape)

    return(arr)


def auto_cor1d(arr, qvalue):
    timesteps = arr.shape[0]
    particles = arr.shape[1]
    j = np.zeros(timesteps,dtype = 'complex_')
    vel = np.empty((timesteps,particles),dtype = 'complex_')
    correlation = np.empty(timesteps)
    radius = np.empty((timesteps,particles),dtype = 'complex_')

    for i in range(timesteps):
        for ii in range(particles):
            vel[i][ii] = arr[i,ii,3]
            radius[i][ii] = arr[i,ii,1]
        j[i] = sum(vel[i]*np.exp(1j*radius[i]*qvalue))/particles
    print(qvalue)
    #print(vel)
    #print(j)
    #j = np.real(j)

    #fig, ax = plt.subplots()
    #x = range(j.shape[0])
    #plt.plot(x, np.real(j), 'g--')
    #plt.show()
    return acf(np.real(j),nlags = 100)



def auto_cor2d(arr, qvalue):
    timesteps = arr.shape[0]
    particles = arr.shape[1]
    j = np.zeros(timesteps,dtype = 'complex_')
    vel = np.empty((timesteps,particles),dtype = 'complex_')
    correlation = np.empty(timesteps)
    radius = np.empty((timesteps,particles),dtype = 'complex_')

    for i in range(timesteps):
        for ii in range(particles):
            vel[i][ii] = math.hypot(arr[i,ii,3], arr[i,ii,4])
            radius[i][ii] = math.hypot(arr[i,ii,1], arr[i,ii,2])
        j[i] = sum(vel[i]*np.exp(1j*radius[i]*qvalue))/particles
        print(i)
    #print(vel)
    #print(j)
    #j = np.real(j)

    #fig, ax = plt.subplots()
    #x = range(j.shape[0])
    #plt.plot(x, np.real(j), 'g--')
    #plt.show()
    return acf(np.real(j))

    #for i in range(timesteps):
        #correlation = np.correlate(j, j, mode = 'full')
    #print(result.size)
    #return np.real(correlation[timesteps//2:]/timesteps)

def auto_cor3d(arr, qvalue):
    timesteps = arr.shape[0]
    particles = arr.shape[1]
    j = np.zeros(timesteps,dtype = 'complex_')
    vel = np.empty((timesteps,particles),dtype = 'complex_')
    correlation = np.empty(timesteps)
    radius = np.empty((timesteps,particles),dtype = 'complex_')

    for i in range(timesteps):
        for ii in range(particles):
            vel[i][ii] = np.sqrt(np.sum(np.array([arr[i,ii,4]*arr[i,ii,4], arr[i,ii,5]*arr[i,ii,5], arr[i,ii,6]*arr[i,ii,6]])))
            radius[i][ii] = np.sqrt(np.sum(np.array([arr[i,ii,1]*arr[i,ii,1], arr[i,ii,2]*arr[i,ii,2], arr[i,ii,3]*arr[i,ii,3]])))
        j[i] = sum(vel[i]*np.exp(1j*radius[i]*qvalue))/particles
        print(i)
    #print(vel)
    #print(j)
    #j = np.real(j)

    #fig, ax = plt.subplots()
    #x = range(j.shape[0])
    #plt.plot(x, np.real(j), 'g--')
    #plt.show()
    return acf(np.real(j))



cut = 50000
timesteps = 200000
hat = 9
particles = 400
line = 7
filename = "test1d.lammpstrj"
#fig, ax = plt.subplots()
#load_data(filename, line, particles)
arrqvalue = np.arange(0.01,2.0,0.005)

for k in arrqvalue:
    y=auto_cor1d(read_data(filename, particles, line, timesteps, hat,cut),k)
###y=auto_cor2d(read_data(filename, particles, line, timesteps, hat,cut),1)
###y=auto_cor3d(read_data(filename, particles, line, timesteps, hat,cut),1.5)
#print(y.shape[0])
#ax.plot(range(y.shape[0]),y)

    outfile="1d/autocor"+str(k)

    np.savetxt(outfile,y)
###plt.show()


#print(arr)

#j_x = np.array(load_data(name)[:,3])
#j_y = np.array(load_data(name)[:,4])
