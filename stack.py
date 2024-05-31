from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import constants as c
import astropy.units as u


h=cosmo.H0.value/100
fac = c.c**2 / (4 * np.pi * c.G)
fac = fac.to(u.Msun * u.pc**(-2)*u.Mpc).value
comoving = True


Mpc_range = 2  # center +- Mpc_range
Mpc_step = 0.3
delta = 4 # pre-selection

xbins = np.linspace(-Mpc_range, Mpc_range, int(2*Mpc_range / Mpc_step))
ybins = np.linspace(-Mpc_range, Mpc_range, int(2*Mpc_range / Mpc_step))


z_sp = 0.1
sourcedata = Table.from_pandas(pd.read_csv('/data/cjia/cluster/sourcewmap3',delim_whitespace=True,names=['ra','dec','e1','e2','res','er1','er2','zs','ds','spa']))
ra0 = sourcedata['ra']
dec0 = sourcedata['dec']


def get_grid_idxs(ra, dec, rac, decc, z, phi):
    dx = (ra - rac) * np.cos(decc / 180 * np.pi)  # degree
    dy = dec - decc                               # degree
    
    scale_d2Mpc = cosmo.angular_diameter_distance(z).value * h / 180 * np.pi
    scale_Mpc2d = 1 / scale_d2Mpc
    
    d_range = Mpc_range * scale_Mpc2d
    d_step = Mpc_step * scale_Mpc2d
    
    grid_list = []
    
    for i in range(len(xbins) - 1):
        for j in range(len(ybins) - 1):
            x1 = xbins[i]
            x2 = xbins[i+1]
            y1 = ybins[j]
            y2 = ybins[j+1]
            
            if (270 < phi <= 360) or (0 <= phi < 90):
                mask = (dy < np.tan(phi / 180 * np.pi) * dx - x1 / np.cos(phi / 180 * np.pi))
                mask = mask * (dy > np.tan(phi / 180 * np.pi) * dx - x2 / np.cos(phi / 180 * np.pi))
                mask = mask * (dx > -np.tan(phi / 180 * np.pi) * dy + y1 / np.cos(phi / 180 * np.pi))
                mask = mask * (dx < -np.tan(phi / 180 * np.pi) * dy + y2 / np.cos(phi / 180 * np.pi))
            elif (90 < phi < 270):
                mask = (dy > np.tan(phi / 180 * np.pi) * dx - x1 / np.cos(phi / 180 * np.pi))
                mask = mask * (dy < np.tan(phi / 180 * np.pi) * dx - x2 / np.cos(phi / 180 * np.pi))
                mask = mask * (dx < -np.tan(phi / 180 * np.pi) * dy + y1 / np.cos(phi / 180 * np.pi))
                mask = mask * (dx > -np.tan(phi / 180 * np.pi) * dy + y2 / np.cos(phi / 180 * np.pi))
            elif phi == 90:
                mask = (dx > x1) * (dx < x2) * (dy > y1) * (dy < y2)
            else:
                mask = (dx < -x1) * (dx > -x2) * (dy < -y1) * (dy > -y2)
            grid_list.append([(x1+x2)/2, (y1+y2)/2, mask])
    
    return grid_list
# return [xMpc, yMpc, idlist(mask)] for each grid





def pre_select(rac, decc, delta):
    premask = (ra0 < rac + delta) * (ra0 > rac - delta) * (dec0 < decc + delta) * (dec0 > decc - delta)
    ra = ra0[premask]
    dec = dec0[premask]
    subsample = sourcedata[premask]
    return [ra, dec, subsample]




def grid_shear(subsample, mask, z_l, phi):
    mask = mask * (subsample['zs'] > z_l + z_sp)
    grid_s = subsample[mask]
    num = len(grid_s)
    if num == 0:
        return [0, 0, 0]
    
    ds = cosmo.angular_diameter_distance(grid_s['zs']).value*h
    dl = cosmo.angular_diameter_distance(z_l).value*h
    dls = cosmo.angular_diameter_distance_z1z2(z_l, grid_s['zs']).value*h
    
    if comoving:
        Sig = fac * ds / (dl * dls) / (1 + z_l)**2
    else:
        Sig = fac * ds / (dl * dls)
    
    wt = 1.0 / (0.1648 + grid_s['er1']**2 + grid_s['er2']**2)    ####   ?
    w = wt / grid_s['res']
    
    theta = - 2 * (phi - 90)
    e1 = np.cos(theta / 180 * np.pi) * grid_s['er1'] - np.sin(theta / 180 * np.pi) * grid_s['er1']
    e2 = np.sin(theta / 180 * np.pi) * grid_s['er1'] + np.cos(theta / 180 * np.pi) * grid_s['er1']
    
    g1 = np.sum(grid_s['e1'] * Sig * w)
    g2 = np.sum(grid_s['e2'] * Sig * w)
    
    return [g1, g2, num]




def stack(clusters):
    dx, dy = clusters[0][:2]
    g1, g2, num = [], [], []
    
    for cluster in clusters:
        g1.append(np.array(cluster[2]))
        g2.append(np.array(cluster[3]))
        num.append(np.array(cluster[4]))
    
    g1 = np.array(g1).T
    g2 = np.array(g2).T
    num = np.array(num).T
    
    g1stk = [np.sum(g1[i] * num[i]) / np.sum(num[i]) for i in range(len(num))]
    g2stk = [np.sum(g2[i] * num[i]) / np.sum(num[i]) for i in range(len(num))]
    numstk = [np.sum(num[i]) for i in range(len(num))]
    
    return [dx, dy, g1stk, g2stk, numstk]





# input
input_data = np.loadtxt('/home/cjia/cluster/lensing/map/input_test', unpack=True)
raclist = input_data[0]
decclist = input_data[1]
zlist = input_data[2]
philist = input_data[4]
clusters = []
for k in range(len(zlist)):
    rac = raclist[k]
    decc = decclist[k]
    z_l = zlist[k]
    phi = philist[k]
    preselect = pre_select(rac, decc, delta)
    ra = preselect[0]
    dec = preselect[1]
    subsample = preselect[2]
    
    dx, dy, g1, g2, num = [], [], [], [], []
    grid_list = get_grid_idxs(ra, dec, rac, decc, z_l, phi)

    for grid in grid_list:
        dx.append(grid[0])
        dy.append(grid[1])
        mask = grid[2]
        shear = grid_shear(subsample, mask, z_l, phi)
        g1.append(shear[0])
        g2.append(shear[1])
        num.append(shear[2])

    clusters.append([dx, dy, g1, g2, num])

result = stack(clusters)

f = open('stack.txt', 'w+')
print('# dx[Mpc] dy[Mpc] g1 g2 num', file=f)
for k in range(len(result[0])):
    print(result[0][k], result[1][k], result[2][k], result[3][k], result[4][k], file=f)
f.close()


