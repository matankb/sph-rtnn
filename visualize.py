import numpy as np
import math as m
import sys,os
import matplotlib.pyplot as plt

TF_HOME = "/home/mkotlerb/"
SPH_HOME = TF_HOME + "sph/" #original-sph/triforce-dev/sph/"

SAVE_FIGS = False
SAVE_FIG_DIR = SPH_HOME + "postprocessing/imgs/"

TF_NAVI = TF_HOME + "tfnavi"
sys.path.append(TF_NAVI)
# from HDF5Reader import *
# import TFNavi as navi
# import PlottingFuncs

# plt.style.use(TF_HOME + "students/mlavell/code/misc/flemch.mplstyle")
good_colors=["#db6d00","#006ddb","#920000","#52a736","#9B30FF"]

fs = 22 #fontsize

# sph data
tf_data_path = SPH_HOME + "data/points-manual/particles_2000.csv"
# tf_data_path = SPH_HOME + "data/temp/particles_500.csv"

pid, x, y, z, vx, vy, vz, mass, rho, p, u, cs, hsml = \
 np.genfromtxt(tf_data_path,skip_header=1,unpack=True,delimiter=',')

color_tf = good_colors[1]
marker_tf = 'o'

# plot
dim=1

if (dim==1):
    r = x
    v = vx
if (dim==2):
    r = (x**2 + y**2)**0.5
    v = (vx**2 + vy**2)**0.5
elif (dim==3):
    r = (x**2 + y**2 + z**2)**0.5
    v = (vx**2 + vy**2 + vz**2)**0.5    
    
print(f"maxr={max(r)}  maxx={max(x)}  maxy={max(y)}")
fig,ax = plt.subplots(2,3,figsize=(9,7))

ax[0,0].set_title("Density",fontsize=fs)
ax[0,0].scatter(r,rho,marker=marker_tf,color=color_tf)

ax[0,1].set_title("Velocity",fontsize=fs)
ax[0,1].scatter(r,v,marker=marker_tf,color=color_tf,label="TFL")

ax[0,2].set_title("Pressure",fontsize=fs)
ax[0,2].scatter(r,p,marker=marker_tf,color=color_tf)

ax[1,0].set_title("Sound speed",fontsize=fs)
ax[1,0].scatter(r,cs,marker=marker_tf,color=color_tf)

ax[1,1].set_title("Internal energy",fontsize=fs)
ax[1,1].scatter(r,u,marker=marker_tf,color=color_tf)

ax[1,2].set_title("Smoothing length",fontsize=fs)
ax[1,2].scatter(r,hsml,marker=marker_tf,color=color_tf)


yorick_data_path = "yorick_diffusion_output.txt"

x_yor, vx_yor, mass_yor, rho_yor, p_yor, u_yor, cs_yor, hsml_yor = \
 np.genfromtxt(yorick_data_path,skip_header=1,unpack=True)

color_yor = good_colors[2]
marker_yor = 'd'

plot_yorick = True

if plot_yorick:
    mew=1.25
    ax[0,0].scatter(x_yor,rho_yor,marker=marker_yor,color=color_yor,facecolors='none',linewidths=mew)
    ax[0,1].scatter(x_yor,vx_yor,marker=marker_yor,color=color_yor,facecolors='none',linewidths=mew,label="Yorick")
    ax[0,2].scatter(x_yor,p_yor,marker=marker_yor,color=color_yor,facecolors='none',linewidths=mew)
    ax[1,0].scatter(x_yor,cs_yor,marker=marker_yor,color=color_yor,facecolors='none',linewidths=mew)
    ax[1,1].scatter(x_yor,u_yor,marker=marker_yor,color=color_yor,facecolors='none',linewidths=mew)
    ax[1,2].scatter(x_yor,hsml_yor,marker=marker_yor,color=color_yor,facecolors='none',linewidths=mew)

    ax[0,1].legend()


[[axis.grid() for axis in axs] for axs in ax]
[axis.set_xlabel("x") for axis in ax[1]]

fig.tight_layout(pad=0.5, rect=[0,0,1,1])
plt.savefig(SAVE_FIG_DIR+"with yorik sph_diffusion.pdf")