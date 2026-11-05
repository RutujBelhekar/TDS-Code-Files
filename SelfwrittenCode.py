import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np



m=1
n=200
inirad=4.2
softcore=0.35
G=4.0 
Krep=0.005
dt=0.0001


np.random.seed(67)


r=inirad*np.sqrt(np.random.rand(n))
theta=np.random.uniform(0,2*np.pi,n)
x=r*np.cos(theta)
y=r*np.sin(theta)

pos=np.column_stack((x,y))
v=np.zeros((n,2))
a=np.zeros((n,2))

i_idx,j_idx=np.tril_indices(n,k=-1)

def forces(pos):
    diff=pos[i_idx]-pos[j_idx]
    dsq=(diff**2).sum(axis=1)+softcore**2
    d=np.sqrt(dsq)
    Fgrav=G*m*m/d**2
    Frep=Krep/d**12
    Fnet=Frep-Fgrav
    Fvec=(diff/d[:,None])*Fnet[:,None]
    Ftot=np.zeros((n,2))
    np.add.at(Ftot,i_idx,Fvec)
    np.add.at(Ftot,j_idx,-Fvec)
    anet=Ftot/m
    return anet,Ftot

anet,Fnet=forces(pos)

for frames in range (1000000):

    v+=anet*0.5*dt
    pos+=v*dt
    anet,Fnet=forces(pos)
    v+=anet*0.5*dt
    if frames % 100000 == 0:
        print(f"Frame {frames} / 1000000 calculated...")

print("Simulation complete! Rendering snapshot...")
fig,ax= plt.subplots(figsize=(10,10))
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)

scat=ax.scatter(pos[:,0],pos[:,1])
plt.show()















#  if frames % 10 == 0:
#       scat.set_offsets(pos)
#        fig.canvas.draw_idle()
#        fig.canvas.flush_events()
