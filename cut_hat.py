import numpy as np

f=open("test.lammpstrj","r")

timesteps = 2001
particles = 10000
hat = 9
line = 5

arr=np.empty(((timesteps-1,particles,line)))
#print(arr[1][1])
for i in range(timesteps-1):
    for ii in range(hat):
        f.readline()
    for ii in range(particles):
        arr[i][ii]=f.readline().split(' ')
print(arr.shape)
np.savetxt("arr.csv", arr.reshape(arr.shape[0],-1))
