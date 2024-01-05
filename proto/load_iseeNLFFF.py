# https://hinode.isee.nagoya-u.ac.jp/nlfff_database/codes/load_nlfff.py


import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/tmp/12192_20141024_170000.nc"
#print(filename)
nc=netCDF4.Dataset(filename,'r')
data = {}
info = {}
for key in ["x", "y", "z"]:
    tmp = nc.variables[key]
    data[key]=tmp[:]
    #print(data[key])
    info[key] = f"{tmp.long_name}  unit: {tmp.units}"

for key in ["Bx", "By", "Bz", "Bx_pot", "By_pot", "Bz_pot"]:
    tmp = nc.variables[key]
    data[key]=tmp[:].transpose(2,1,0)
    #print(data[key])
    info[key] = f"{tmp.long_name}  unit: {tmp.units}"

print(info)

fig=plt.figure()
ax = fig.add_subplot()#projection='3d')
im = ax.pcolormesh(data["x"],data["y"],data["Bz"][:,:,0].transpose(),cmap='gist_gray',shading='auto')
plt.show()
