import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np



m=1
n=100
inirad=5.0
softcore=0.35
G=0.5 
Krep=0.0001
dt=0.001


np.random.seed(67)


r=inirad*np.sqrt(np.random.rand(n))
theta=np.random.uniform(0,2*np.pi,n)
x=r*np.cos(theta)
y=r*np.sin(theta)

pos=np.column_stack((x,y))
v=np.zeros((n,2))
a=np.zeros((n,2))

i_idx,j_idx=np.tril_indices(n,k=-1)

def forces(x,y):
    dx=x[i_idx]-x[j_idx]
    dy=y[i_idx]-y[j_idx]
    dsq=(dx*dx+dy*dy+softcore**2)
    d=np.sqrt(dsq)
    Fgrav=G*m*m/d**2
    Frep=Krep/d**12
    Fnet=Frep-Fgrav
    fx=(Fnet*dx/d).sum(axis=1)
    fy=(Fnet*dy/d).sum(axis=1)
    ax=fx/m
    ay=fy/m
    return fx,fy,ax,ay

plt.ion()
fig,ax= plt.subplots(figsize=(10,10))
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)

scat=ax.scatter(x,y)

_,_,ax,ay=forces(x,y)




