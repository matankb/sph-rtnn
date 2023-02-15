import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
matplotlib.use('qt5agg')


# def animate_contour(xdata=None, ydata=None, zdata=None, times=None, timescale='s', title='', xlabel='', ylabel='',
#                     cmap='viridis', zlims=None, levels=100, save=False, filename=None):
#     nt = len(zdata)
#
#     if xdata is None:
#         m = zdata[0].shape[1]
#         xdata = [i for i in range(m)]
#     if ydata is None:
#         n = zdata[0].shape[0]
#         ydata = [j for j in range(n)]
#
#     vmin = zlims[0] if zlims is not None else np.min(zdata)
#     vmax = zlims[1] if zlims is not None else np.max(zdata)
#
#     norm = Normalize(vmin=vmin, vmax=vmax)
#
#     if times is None:
#         title_str = title + f'\n0/{nt}'
#     else:
#         title_str = title + f'\n{times[0]:4.3f}{timescale}'
#
#     fig, ax = plt.subplots(1, 1)
#     fig.set_tight_layout(True)
#     fig_title = ax.set_title(title_str)
#
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#
#     cax = make_axes_locatable(ax).append_axes('right', '5%', '5%')
#
#     im = ax.contourf(xdata, ydata, zdata[0], cmap=cmap, levels=levels, norm=norm)
#
#     if zlims is not None:
#         fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
#
#     def animate(i):
#         nonlocal im, fig, cax
#         for c in im.collections:
#             c.remove()
#
#         if zlims is None:
#             im = ax.contourf(xdata, ydata, zdata[i], cmap=cmap, levels=levels)
#             cax.clear()
#             fig.colorbar(im, cax=cax)
#         else:
#             im = ax.contourf(xdata, ydata, zdata[i], cmap=cmap, levels=levels, norm=norm)
#
#         if times is None:
#             title_str = title + f'\n{i}/{nt}'
#         else:
#             title_str = title + f'\n{times[i]:4.3f}{timescale}'
#
#         fig_title.set_text(title_str)
#         return im.collections
#
#     anim = FuncAnimation(fig, animate, frames=len(zdata), interval=50, blit=False)
#
#     if save:
#         anim.save(filename, writer='ffmpeg')
#         print(f'Animation saved as {filename}.')
#     else:
#         plt.show()
#
#
# def animate_line(arrs, title='', xlabel='', ylabel='', xlims=None, ylims=None, save=False, filename=None):
#     nt = len(arrs)

#     fig, ax = plt.subplots(1, 1)

#     title = ax.set_title(f'Frame 0 / {nt}')
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

#     if ylims is not None:
#         ax.set_ylim(ylims[0], ylims[1])
#     else:
#         ax.set_ylim(np.min(arrs), np.max(arrs))

#     line, = ax.plot(arrs[0])

#     def animate(i):
#         line.set_ydata(arrs[i])

#         title.set_text(f'Frame {i} / {nt}')
#         return line,

#     anim = FuncAnimation(fig, animate, frames=len(arrs), interval=100, blit=False)

#     if save:
#         anim.save(filename, writer='ffmpeg')
#         print(f'Animation saved as {filename}.')
#     else:
#         plt.show()


def animate_diffusion(xarrs, yarrs, title='', xlabel='', ylabels='', xlims=None, ylims=None, save=False, filename=None):
    nt = len(xarrs)
    narrs =  len(yarrs)

    fig, axs = plt.subplots(narrs, 1, figsize=(6,8))

    title = fig.suptitle(f'Frame 0 / {nt}')

    lines = []

    for i in range(narrs):    
        ax = axs[i]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabels[i])
        
        if ylims is not None:
            ax.set_ylim(ylims[i][0], ylims[i][1])
        else:
            ax.set_ylim(np.min(yarrs[i]), np.max(yarrs[i]))

        line, = ax.plot(xarrs[0], yarrs[i][0])
        lines.append(line)

    def animate(i):
        for j in range(narrs):
            lines[j].set_xdata(xarrs[i])
            lines[j].set_ydata(yarrs[j][i])

            axs[j].set_xlim(np.min(xarrs[i]),np.max(xarrs[i]))

        fig.suptitle(f'Frame {i} / {nt}')
        fig.tight_layout(pad=0.4, rect=[0,0,1,0.95])

        return lines,

    anim = FuncAnimation(fig, animate, frames=nt, interval=0.1, blit=False)

    if save:
        anim.save(filename, writer='ffmpeg')
        print(f'Animation saved as {filename}.')
    else:
        plt.show()


def animate_line(xarrs, arrs, title='', xlabel='', ylabel='', xlims=None, ylims=None, save=False, filename=None):
    nt = len(arrs)

    fig, ax = plt.subplots(1, 1)

    title = ax.set_title(f'Frame 0 / {nt}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylims is not None:
        ax.set_ylim(ylims[0], ylims[1])
    else:
        ax.set_ylim(np.min(arrs), np.max(arrs))

    line, = ax.plot(xarrs[0], arrs[0])

    def animate(i):
        line.set_xdata(xarrs[i])
        line.set_ydata(arrs[i])

        ax.set_xlim(np.min(xarrs[i]),np.max(xarrs[i]))

        title.set_text(f'Frame {i} / {nt}')
        return line,

    anim = FuncAnimation(fig, animate, frames=len(arrs), interval=100, blit=False)

    if save:
        anim.save(filename, writer='ffmpeg')
        print(f'Animation saved as {filename}.')
    else:
        plt.show()


def compute_particle_radius(positions):
    return [np.sum(frame**2, axis=1) for frame in positions]


def get_positions(filepath):
    positions = []
    dir_files = sorted(os.listdir(filepath), key=lambda x: int(x[:-4].split('_')[-1]))
    for file in dir_files:
        df = pd.read_csv(f'{filepath}/{file}', dtype=np.float64, skipinitialspace=True)
        positions.append(df[['x', 'y', 'z']].to_numpy())

    return positions

def get_densities(filepath):
    densities = []
    dir_files = sorted(os.listdir(filepath), key=lambda x: int(x[:-4].split('_')[-1]))
    for file in dir_files:
        df = pd.read_csv(f'{filepath}/{file}', dtype=np.float64, skipinitialspace=True)
        densities.append(df[['rho']].to_numpy())

    return densities

def get_pressures(filepath):
    pressures = []
    dir_files = sorted(os.listdir(filepath), key=lambda x: int(x[:-4].split('_')[-1]))
    for file in dir_files:
        df = pd.read_csv(f'{filepath}/{file}', dtype=np.float64, skipinitialspace=True)
        pressures.append(df[['pressure']].to_numpy())

    return pressures


def get_velocities(filepath):
    velocities = []
    dir_files = sorted(os.listdir(filepath), key=lambda x: int(x[:-4].split('_')[-1]))
    for file in dir_files:
        df = pd.read_csv(f'{filepath}/{file}', dtype=np.float64, skipinitialspace=True)
        velocities.append(df[['vx', 'vy', 'vz']].to_numpy())

    return velocities


def get_internal_energy(filepath):
    energies = []
    dir_files = sorted(os.listdir(filepath), key=lambda x: int(x[:-4].split('_')[-1]))
    for file in dir_files:
        df = pd.read_csv(f'{filepath}/{file}', dtype=np.float64, skipinitialspace=True)
        energies.append(df['internal energy'].to_numpy())

    return energies


def main():
    dpath = './data/Diffusion'
    positions    = get_positions(dpath)
    velocities   = get_velocities(dpath)
    densities    = get_densities(dpath)
    pressures    = get_pressures(dpath)

    eps_internal = get_internal_energy(dpath)
    # radius = compute_particle_radius(positions)

    nsteps = np.shape(densities)[0]

    xpos = [positions[p][:,0] for p in range(nsteps) ]
    xvel = [velocities[v][:,0] for v in range(nsteps) ]

    # animate_line(px,densities,save=False,filename='density1d.mp4')

    xlabel = 'position'
    yarrs = [densities,xvel,pressures,eps_internal]
    ylabels = ['density','velocity','pressure','internal energy']
    animate_diffusion(xpos,yarrs,xlabel=xlabel,ylabels=ylabels)

if __name__ == '__main__':
    main()
