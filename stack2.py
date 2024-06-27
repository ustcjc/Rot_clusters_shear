from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import constants as c
import astropy.units as u
import multiprocessing
import time


h=cosmo.H0.value/100
fac = c.c**2 / (4 * np.pi * c.G)
fac = fac.to(u.Msun * u.pc**(-2)*u.Mpc).value
comoving = True


Mpc_range = 1  # center +- Mpc_range
Mpc_step = 0.1
delta = 2 # pre-selection

xbins = np.linspace(-Mpc_range, Mpc_range, int(2*Mpc_range / Mpc_step))
ybins = np.linspace(-Mpc_range, Mpc_range, int(2*Mpc_range / Mpc_step))


# UNIONS data
data = fits.open('/data/cjia/cluster/unions_shapepipe_2022_v1.3.fits')[1].data
ra = data['RA']
dec = data['Dec']
e1 = data['e1']
e2 = data['e2']
w = data['w']

'''
# SDSS data
sourcedata = Table.from_pandas(pd.read_csv('/data/cjia/cluster/sourcewmap3',delim_whitespace=True,names=['ra','dec','e1','e2','res','er1','er2','zs','ds','spa']))
ra = sourcedata['ra']
dec = sourcedata['dec']
e1 = sourcedata['e1']
e2 = sourcedata['e2']
w = 1 / (1 + sourcedata['er1']**2 + sourcedata['er2']**2)
'''




def get_grid_idxs(ra, dec, rac, decc, z, phi):
    dx = - (ra - rac) * np.cos(decc / 180 * np.pi)  # degree
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
            grid_list.append([(x1+x2)/2, (y1+y2)/2, mask, i * (len(ybins)-1) + j])
    
    return grid_list
# return [xMpc, yMpc, idlist(mask)] for each grid





def pre_select(rac, decc, delta):
    premask = (ra < rac + delta) * (ra > rac - delta) * (dec < decc + delta) * (dec > decc - delta)
    rasub = ra[premask]
    decsub = dec[premask]
    e1sub = e1[premask]
    e2sub = e2[premask]
    wsub = w[premask]
    return [rasub, decsub, e1sub, e2sub, wsub]




def grid_shear(e1sub, e2sub, wsub, mask, phi, dx, dy, gridid):
    e1 = e1sub[mask]
    e2 = e2sub[mask]
    w = wsub[mask]
    num = len(w)
    if num == 0:
        return [0, 0, 0, 0, 0, gridid]
    
    theta = - 2 * (phi - 90)
    e1rot = np.cos(theta / 180 * np.pi) * e1 - np.sin(theta / 180 * np.pi) * e2
    e2rot = np.sin(theta / 180 * np.pi) * e1 + np.cos(theta / 180 * np.pi) * e2
    
    g1 = np.sum(e1rot * w) / num
    g2 = np.sum(e2rot * w) / num
    
    return [dx, dy, g1, g2, num, gridid]





def stack(clusters):
    dx_all = clusters[0]
    dy_all = clusters[1]
    g1_all = clusters[2]
    g2_all = clusters[3]
    num_all = clusters[4]
    gridid = clusters[5]
    dx, dy, g1, g2, num = [], [], [], [], []
    for k in range((len(xbins)-1) * (len(ybins)-1)):
        mask = gridid == k
        if ((np.abs(dx_all[mask][0]) + np.abs(dy_all[mask][0])) != 0):
            dx.append(dx_all[mask][0])
            dy.append(dy_all[mask][0])
            g1.append(np.sum(g1_all[mask] * num_all[mask]) / np.sum(num_all[mask]))
            g2.append(np.sum(g2_all[mask] * num_all[mask]) / np.sum(num_all[mask]))
            num.append(np.sum(num_all[mask]))
    return [dx, dy, g1, g2, num]
        




# input
input_data = np.loadtxt('/home/cjia/cluster/lensing/non_rot/non_rot_clusters.txt', unpack=True)
raclist = input_data[1]
decclist = input_data[2]
zlist = input_data[3]
philist = input_data[4]
clusters = []
for k in range(len(zlist)):
#for k in range(3):
    start_time1 = time.time()
    print(k)  #
    rac = raclist[k]
    decc = decclist[k]
    z_l = zlist[k]
    phi = philist[k]
    preselect = pre_select(rac, decc, delta)
    rasub = preselect[0]
    decsub = preselect[1]
    e1sub = preselect[2]
    e2sub = preselect[3]
    wsub = preselect[4]
    
    start_time = time.time()
    grid_list = get_grid_idxs(rasub, decsub, rac, decc, z_l, phi)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Run: {execution_time} s')
    
    for i in range(len(xbins) - 1):
        calculation = multiprocessing.Pool(processes=len(ybins) - 1)
        for j in range(len(ybins) - 1):
            grid = grid_list[i * (len(ybins)-1) + j]
            calculation.apply_async(grid_shear, (e1sub, e2sub, wsub, grid[2], phi, grid[0], grid[1], grid[3],), callback=clusters.append)
            
        calculation.close()
        calculation.join()
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1
    print(f'WholeRun: {execution_time1} s')

clusters = np.array(clusters).T
print(clusters.shape)

result = stack(clusters)

f = open('stack.txt', 'w+')
print('# dx[Mpc] dy[Mpc] g1 g2 num', file=f)
for k in range(len(result[0])):
    print(result[0][k], result[1][k], result[2][k], result[3][k], result[4][k], file=f)
f.close()


