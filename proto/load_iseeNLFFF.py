# https://hinode.isee.nagoya-u.ac.jp/nlfff_database/codes/load_nlfff.py


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
from matplotlib import colormaps as cmp
from progressbar import progressbar
import netCDF4
import sys
import os

import line2obj

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
print(np.shape(data["Bz"]))
print(np.shape(data["x"]), np.shape(data["y"]), np.shape(data["z"]))
fig=plt.figure()
ax = fig.add_subplot()#projection='3d')
im = ax.pcolormesh(data["x"],data["y"],np.arcsinh(data["Bz"][:,:,0].transpose()),cmap='RdBu',shading='auto')
plt.colorbar(im)
plt.show()

def getVectInterp(vx, vy, vz, idx, idy, idz):
    maxx,maxy,maxz = np.shape(vx)
    idx_l = int(idx)
    idx_h = np.max([idx_l + 1,maxx-1])
    idy_l = int(idy)
    idy_h = np.max([idy_l + 1,maxy-1])
    idz_l = int(idz)
    idz_h = np.max([idz_l + 1,maxz-1])
    wxh = idx - idx_l
    wxl = 1.0 - wxh
    wyh = idy - idy_l
    wyl = 1.0 - wyh
    wzh = idz - idz_l
    wzl = 1.0 - wzh
    vct = np.zeros(3)
    for ix, wx in zip([idx_l,idx_h],[wxl,wxh]):
        for iy, wy in zip([idy_l,idy_h],[wyl,wyh]):
            for iz, wz in zip([idz_l,idz_h],[wzl,wzh]):
                vct[0] += wx*wy*wz*vx[ix,iy,iz]
                vct[1] += wx*wy*wz*vy[ix,iy,iz]
                vct[2] += wx*wy*wz*vz[ix,iy,iz]
    return vct

def gettrace(vx,vy,vz, idx, idy, idz, direct="both", sd=1, dd=1):
    if direct != "forward" or direct != "backward":
        direct = "both"
    path = [[float(idx), float(idy), float(idz)]]
    dircoef = []
    if direct == "forward" or direct == "both":
        dircoef.append(float(dd))
    if direct == "backward" or direct == "both":
        dircoef.append(-1.0*dd)
    savelen = float(sd)/dd
    maxx,maxy,maxz = np.shape(vx)
    maxx -= 1
    maxy -= 1
    maxz -= 1
    for dc in dircoef:
        cx, cy, cz = float(idx), float(idy), float(idz)
        tmppath = []
        len = 0
        for cnt in range(10000):
            vct = getVectInterp(vx,vy,vz,cx,cy,cz)
            vct = dc * vct  / (vct**2).sum()**0.5
            nx = cx + vct[0]
            ny = cy + vct[1]
            nz = cz + vct[2]
            if nx<0 or nx>=maxx or ny<0 or ny>=maxy or nz<0 or nz>=maxz:
                tmppath.append(np.array([nx,ny,nz]))
                break
            cx, cy, cz = nx, ny, nz
            len += 1
            if len >= savelen:
                tmppath.append(np.array([cx,cy,cz]))
                len = 0
        if dd < 0:
            path = tmppath[1::-1] + path
        else:
            path = path + tmppath[1:]
    return path

pts = []
bbmax = np.max(data["Bx"]**2 + data["By"]**2+data["Bz"]**2)
for ii in progressbar(np.arange(500)):
    for jj in np.arange(1000):
        idx = 100+400*np.random.rand()
        idy = 50+100*np.random.rand()
        tmpb = getVectInterp(data["Bx"],data["By"],data["Bz"],idx,idy,0)
        bb = tmpb[0]**2 + tmpb[1]**2+tmpb[2]**2
        if bb >= bbmax*np.random.rand(): # and tmpb[2]>0 :
            pts.append([idx,idy,0])
            break
paths = []
for pt in progressbar(pts):
    #paths.append(gettrace(data["Bx"],data["By"],data["Bz"],*pt, dd=1)) # bad
    #paths.append(gettrace(data["Bx"],data["Bz"],data["By"],*pt, dd=0.1)) # ?
    #paths.append(gettrace(data["By"],data["Bx"],data["Bz"],*pt, dd=1)) # bad
    #paths.append(gettrace(data["By"],data["Bz"],data["Bx"],*pt, dd=1)) # bad
    paths.append(gettrace(data["Bz"],data["Bx"],data["By"],*pt, dd=0.1)) # ?
    #paths.append(gettrace(data["Bz"],data["By"],data["Bx"],*pt, dd=1)) # bad
#print(paths)
selected = [path for path in paths if len(path)>100]
yy, xx = np.meshgrid(np.arange(len(data["y"])),np.arange(len(data["x"])))
norm = mcl.Normalize(vmin=-8,vmax=8)
cmap = cmp.get_cmap("RdBu")
colors = plt.cm.RdBu(norm(np.arcsinh(data["Bz"][:,:,0])))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.plot_surface(xx,yy,np.zeros_like(xx),facecolors=colors,cstride=10,rstride=10)
for path in selected:
    xs = [pt[0] for pt in path]
    ys = [pt[1] for pt in path]
    zs = [pt[2] for pt in path]
    ax.plot(xs, ys, zs, zorder=100, c=[0.5,0.5,0.0])
plt.show()

line2obj.line2obj(ax, "proto/tmp/noaa12192.obj", "proto/tmp/noaa12192.mtl", radius=0.3)
