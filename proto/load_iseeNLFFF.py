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
sx, sy = np.shape(data["Bz"][:,:,0])
#fig,ax = plt.subplots(1,figsize=(sx/100.0, sy/100.0))

fig=plt.figure(figsize=(sx/100.0, sy/100.0),frameon=False)
ax = fig.add_subplot()#projection='3d')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
ax.set_axis_off()
im = ax.pcolormesh(data["x"],data["y"],np.arcsinh(data["Bx"][:,:,0].transpose()),cmap='RdBu')
plt.show()

fig=plt.figure(figsize=(sx/100.0, sy/100.0),frameon=False)
ax = fig.add_subplot()#projection='3d')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
ax.set_axis_off()
im = ax.pcolormesh(data["x"],data["y"],np.arcsinh(data["By"][:,:,0].transpose()),cmap='RdBu')
plt.show()

fig=plt.figure(figsize=(sx/100.0, sy/100.0),frameon=False)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
ax = fig.add_subplot()#projection='3d')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
#ax.axis('tight')
#ax.axis('off')
ax.set_axis_off()
#ax.set_xmargin(0)
#ax.set_ymargin(0)
im = ax.pcolormesh(data["x"],data["y"],np.arcsinh(data["Bz"][:,:,0].transpose()),cmap='RdBu')#,shading='auto')
#plt.colorbar(im)
fig.savefig("proto/tmp/noaa12192_Bz.png")
plt.show()
print(im)


qx, qz = np.array([[513*np.random.rand(), 257*np.random.rand()] for ii in range(1000)]).transpose()
qvx = np.array([data["Bx"][int(ix),127,int(iz)] for ix, iz in zip(qx, qz)])
qvz = np.array([data["Bz"][int(ix),127,int(iz)] for ix, iz in zip(qx, qz)])
qx = [data["x"][int(ix)] for ix in qx]
qz = [data["z"][int(iz)] for iz in qz]
fig=plt.figure(figsize=(sx/100.0, sy/100.0),frameon=False)
ax = fig.add_subplot()#projection='3d')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
ax.set_axis_off()
im = ax.pcolormesh(data["x"],data["z"],np.arcsinh(data["Bx"][:,127,:].transpose()),cmap='RdBu')
qv = ax.quiver(qx, qz, qvx, qvz)
plt.show()

fig=plt.figure(figsize=(sx/100.0, sy/100.0),frameon=False)
ax = fig.add_subplot()#projection='3d')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
ax.set_axis_off()
im = ax.pcolormesh(data["x"],data["z"],np.arcsinh(data["By"][:,127,:].transpose()),cmap='RdBu')
plt.show()

fig=plt.figure(figsize=(sx/100.0, sy/100.0),frameon=False)
ax = fig.add_subplot()#projection='3d')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
ax.set_axis_off()
im = ax.pcolormesh(data["x"],data["z"],np.arcsinh(data["Bz"][:,127,:].transpose()),cmap='RdBu')
qv = ax.quiver(qx, qz, qvx, qvz)
plt.show()


def getVectInterp(vx, vy, vz, idx, idy, idz):
    maxx,maxy,maxz = np.shape(vx)
    idx_l = int(idx)
    idx_h = np.min([idx_l+1, maxx-1])
    idy_l = int(idy)
    idy_h = np.min([idy_l+1, maxy-1])
    idz_l = int(idz)
    idz_h = np.min([idz_l+1, maxz-1])
    wxh = np.max([0.0, np.min([idx - idx_l, 1.0])])
    wxl = 1.0 - wxh
    wyh = np.max([0.0, np.min([idy - idy_l, 1.0])])
    wyl = 1.0 - wyh
    wzh = np.max([0.0, np.min([idz - idz_l, 1.0])])
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
    if type(dd) != list and type(dd) != np.ndarray:
        dd = [dd]*3
    dd = np.array(dd)
    if direct != "forward" and direct != "backward":
        direct = "both"
    path = [[float(idx), float(idy), float(idz)]]
    drctlst = []
    if direct == "forward" or direct == "both":
        drctlst.append("forward")
    if direct == "backward" or direct == "both":
        drctlst.append("backward")
    savelen = float(sd)/(dd**2).sum()**0.5
    maxx,maxy,maxz = np.shape(vx)
    maxx -= 1
    maxy -= 1
    maxz -= 1
    for drctkey in drctlst:
        dc = dd if drctkey == "forward" else -1*dd
        #print(dc)
        cx, cy, cz = float(idx), float(idy), float(idz)
        tmppath = []
        len = 0
        for cnt in np.arange(savelen*10000):
            vct = getVectInterp(vx,vy,vz,cx,cy,cz)
            #vct = np.zeros(3)
            #vct[0] = vx[int(cx), int(cy), int(cz)]
            #vct[1] = vy[int(cx), int(cy), int(cz)]
            #vct[2] = vz[int(cx), int(cy), int(cz)]
            vct = dc * vct  / (vct**2).sum()**0.5
            #if vct[0] == 0:
            #print(vct)
            nx = cx + vct[0]
            ny = cy + vct[1]
            nz = cz + vct[2]

            if nx<0 or nx>=maxx or ny<0 or ny>=maxy or nz<0 or nz>=maxz:
                tmppath.append(np.array([nx,ny,nz]))
                break
            cx, cy, cz = nx, ny, nz
            #print([vct, [cx, cy, cz]])
            len += 1
            if len >= savelen:
                tmppath.append(np.array([cx,cy,cz]))
                len = 0
        if drctkey == "backward":
            path = tmppath[:1:-1] + path
        else:
            path = path + tmppath[1:]
    return path

pts = []
bbmax = np.max(data["Bx"]**2 + data["By"]**2+data["Bz"]**2)

#pts.append([300,127,0])

for ii in progressbar(np.arange(200)):
    for jj in np.arange(1000):
        idx = 100+300*np.random.rand()
        idy = 50+150*np.random.rand()
        tmpb = getVectInterp(data["Bx"],data["By"],data["Bz"],idx,idy,0)
        bb = tmpb[0]**2 + tmpb[1]**2 + tmpb[2]**2
        if bb >= bbmax*np.random.rand(): # and tmpb[2]>0 :
            pts.append([idx,idy,0])
            break

paths = []
dd = 0.1
drct = np.array([1,1,1], dtype=float)
if data["x"][0] > data["x"][-1]:
    drct[0] = -1
if data["y"][0] > data["y"][-1]:
    drct[1] = -1
if data["z"][0] > data["z"][-1]:
    drct[2] = -1
dd = dd * drct / (drct**2).sum()**0.5
print(f"x: {data['x'][0]}, {data['x'][-1]}")
print(f"y: {data['y'][0]}, {data['y'][-1]}")
print(f"z: {data['z'][0]}, {data['z'][-1]}")
print(dd)
for pt in progressbar(pts):
    paths.append(gettrace(data["Bx"],data["By"],data["Bz"],*pt, dd=dd)) # bad
    #paths.append(gettrace(data["Bx"],data["Bz"],data["By"],*pt, dd=dd)) # ?
    #paths.append(gettrace(data["By"],data["Bx"],data["Bz"],*pt, dd=dd)) # bad
    #paths.append(gettrace(data["By"],data["Bz"],data["Bx"],*pt, dd=dd)) # bad
    #paths.append(gettrace(data["Bz"],data["Bx"],data["By"],*pt, dd=dd)) # ?
    #paths.append(gettrace(data["Bz"],data["By"],data["Bx"],*pt, dd=dd)) # bad

#print(paths)
selected = [] #[path for path in paths if len(path)>10]
for path in paths:
    bb = getVectInterp(data["Bx"],data["By"],data["Bz"],*path[0])
    if path[-1][2] <= 1 and bb[2] < 0: # closed loop
        continue
    selected.append(path)
xx , yy = np.meshgrid(np.arange(len(data["x"])),np.arange(len(data["y"])))
norm = mcl.Normalize(vmin=-8,vmax=8)
cmap = cmp.get_cmap("RdBu")
colors = plt.cm.RdBu(norm(np.arcsinh(data["Bz"][:,:,0].transpose())))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx,yy,np.zeros_like(xx),facecolors=colors,cstride=10,rstride=10)

for path in selected:
    xs = [pt[0] for pt in path]
    ys = [pt[1] for pt in path]
    zs = [pt[2] for pt in path]
    ax.plot(xs, ys, zs, zorder=100, c=[0.5,0.5,0.0])

#ax.axis("equal")
plt.show()
#print(dir(ax))
#print(paths)
#with open("paths.txt", "w") as ofile:
#    ofile.write(str(paths))
line2obj.line2obj(ax, "proto/tmp/noaa12192.obj", "proto/tmp/noaa12192.mtl", radius=0.3)
