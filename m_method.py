import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import random

from matplotlib.ticker import AutoMinorLocator


data = np.loadtxt('/data/cjia/cluster/planck18_galaxy', unpack=True)
gid = data[0] # group ID
ra = data[1]
dec = data[2]
z = data[3]
mag = data[7]
clr = data[8]
bcg = data[9]

data_c = np.loadtxt('/data/cjia/cluster/group_lum_center.dat', unpack=True)
gid_c = data_c[0]
ra_c = data_c[1]
dec_c = data_c[2]
z_c = data_c[3]


c0 = 300000 # km/s
nbin = 91


def get_bcg(group_id):
    mask = gid == group_id
    ra_group = ra[mask]
    dec_group = dec[mask]
    z_group = z[mask]
    bcg_group = bcg[mask]
    
    bcg_index = bcg_group == 1
    
    return [ra_group[bcg_index][0], dec_group[bcg_index][0], z_group[bcg_index][0]]



def calculate_ideal(dx, dy, v_rela, vmax, phi_axis, dv, dverr):
    vlist = v_rela.copy()
    axis_up = (dy >= dx * np.tan(phi_axis * np.pi / 180.0))
    axis_dw = (dy <= dx * np.tan(phi_axis * np.pi / 180.0))
    sign = np.sign(np.tan(phi_axis * np.pi / 180.0)) * np.sign(np.sin(phi_axis * np.pi / 180.0))
    vlist[axis_up] = vmax / 2 * sign
    vlist[axis_dw] = -vmax / 2 * sign
    
    phi = np.linspace(1.0, 361.0, nbin)
    dv_id = [] # <z1> - <z2>
    for i in range(nbin):
        ix_up = (dy >= dx * np.tan(phi[i] * np.pi / 180.0))
        ix_dw = (dy <= dx * np.tan(phi[i] * np.pi / 180.0))
        sign = np.sign(np.tan(phi[i] * np.pi / 180.0)) * np.sign(np.sin(phi[i] * np.pi / 180.0))
        delta_v = (np.mean(vlist[ix_up]) - np.mean(vlist[ix_dw])) * sign
        dv_id.append(delta_v)
    dv_id = np.array(dv_id)
    
    chi2 = np.sum((dv - dv_id)**2 / dverr**2) / nbin
    return [dv_id, chi2]


def calculate_random(dx, dy, v_rela, dv, dverr):
    dv_rand_mean = []
    dv_rand_err = []
    dv_rand_samples = []
    for k in range(10000):
        vlist = v_rela.copy()
        random.shuffle(vlist)
    
        phi = np.linspace(1.0, 361.0, nbin)
        dv_rand = [] # <z1> - <z2>
        for i in range(nbin):
            ix_up = (dy >= dx * np.tan(phi[i] * np.pi / 180.0))
            ix_dw = (dy <= dx * np.tan(phi[i] * np.pi / 180.0))
            sign = np.sign(np.tan(phi[i] * np.pi / 180.0)) * np.sign(np.sin(phi[i] * np.pi / 180.0))
            delta_v = (np.mean(vlist[ix_up]) - np.mean(vlist[ix_dw])) * sign
            dv_rand.append(delta_v)
        dv_rand = np.array(dv_rand)
        dv_rand_samples.append(dv_rand)
    dv_rand_samples = np.transpose(np.array(dv_rand_samples))
    
    for k in range(len(dv_rand_samples)):
        dv_rand_mean.append(np.mean(dv_rand_samples[k]))
        dv_rand_err.append(np.std(dv_rand_samples[k]))
    dv_rand_mean = np.array(dv_rand_mean)
    dv_rand_err = np.array(dv_rand_err)
    
    chi2 = np.sum((dv - dv_rand_mean)**2 / (dverr**2 + dv_rand_err**2)) / nbin
    return [dv_rand_mean, chi2]

    


def run_cluster(gid_run, ra_center, dec_center, z_center):
    z_to_v = c0 / (1 + z_center) # km/s
    
    mask = (gid == gid_run)
    ra_run = ra[mask]
    dec_run = dec[mask]
    z_run = z[mask]
    mag_run = mag[mask]
    clr_run = clr[mask]
    richness = len(z_run)
    gid_run = int(gid_run)
    
    dx = (ra_run - ra_center) * np.cos(dec_run / 180.0 * np.pi)
    dy = dec_run - dec_center
    v_rela = (z_run - z_center) * z_to_v
    
    
    phi = np.linspace(1.0, 361.0, nbin)
    dv = [] # <z1> - <z2>
    dverr = []
    for i in range(nbin):
        ix_up = (dy >= dx * np.tan(phi[i] * np.pi / 180.0))
        ix_dw = (dy <= dx * np.tan(phi[i] * np.pi / 180.0))
        sign = np.sign(np.tan(phi[i] * np.pi / 180.0)) * np.sign(np.sin(phi[i] * np.pi / 180.0))
        delta_v = (np.mean(v_rela[ix_up]) - np.mean(v_rela[ix_dw])) * sign
        error = np.sqrt(np.std(v_rela[ix_up])**2 / len(v_rela[ix_up]) + np.std(v_rela[ix_dw])**2 / len(v_rela[ix_dw]))
        dv.append(delta_v)
        dverr.append(error)
    dv = np.array(dv)
    dverr = np.array(dverr)
    
    vmax = dv.max() #
    index = dv.tolist().index(dv.max())
    verr = dverr[index]
    phi_axis = phi[index] #
    
    ideal = calculate_ideal(dx, dy, v_rela, vmax, phi_axis, dv, dverr)
    dv_id = ideal[0]
    chi2_id = ideal[1]
    
    rand = calculate_random(dx, dy, v_rela, dv, dverr)
    dv_rand = rand[0]
    chi2_rand = rand[1]
    
    
    # Info
    os.system(f'mkdir new/m_info')
    f = open(f'new/m_info/group{gid_run}.txt', 'w+')
    print('# GroupID Vrot[km/s] Verr[km/s] Phi_axis[degree] rdc_chi2_id rdc_chi2_rand', file=f)
    print(gid_run, vmax, verr, phi_axis, chi2_id, chi2_rand, file=f)
    f.close()
    
    
    
    # Plot
    os.system(f'mkdir new/m_rot_curve')
    
    font_label = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 25}
    font_legend = {'family': 'serif', 'weight': 'normal', 'size': 18}
    font_title = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 35}
    
    fig1, axs1 = plt.subplots(figsize=(8, 7), layout='constrained')
    width = 2
    for bound in ['top', 'bottom', 'left', 'right']:
        axs1.spines[bound].set_linewidth(width)
    axs1.plot(phi, dv, c='steelblue', linewidth=2, label='real')
    axs1.plot(phi, dv_id, c='darkorange', linewidth=2, label='ideal')
    axs1.plot(phi, dv_rand, c='m', linewidth=2, label='random')
    axs1.fill_between(phi, dv - dverr, dv + dverr,
                      color='skyblue', alpha=0.2)
    axs1.tick_params(axis='both', direction='in', length=10, width=width, colors='black', labelsize=20)
    axs1.set_xlabel(r'$\phi$ [degree]', fontdict=font_label)
    axs1.set_ylabel(r'$V_{diff}$ [km/s]', fontdict=font_label)
    axs1.set_title(f'Cluster{gid_run}', fontdict=font_label)
    axs1.legend(prop=font_legend)
    fig1.savefig(f'new/m_rot_curve/group{gid_run}.png')
    
    
    os.system(f'mkdir new/m_scatter')
    
    fig2, axs2 = plt.subplots(figsize=(8, 7))
    width = 2
    for bound in ['top', 'bottom', 'left', 'right']:
        axs2.spines[bound].set_linewidth(width)
    figure = axs2.scatter(dx, dy, c=v_rela, cmap='RdBu', s=20, alpha=0.6)
    cmap = figure.get_cmap()
    reversed_cmap = cm.get_cmap(cmap.name + '_r')
    figure.set_cmap(reversed_cmap)
    colorbar = plt.colorbar(figure)
    colorbar.ax.tick_params(labelsize=15)
    colorbar.set_label(r'$V_{r}$ [km/s]', fontsize=15)
    axs2.tick_params(axis='both', direction='in', length=10, width=width, colors='black', labelsize=20)
    axs2.set_xlabel(r'x', fontdict=font_label)
    axs2.set_ylabel(r'y', fontdict=font_label)
    axs2.set_title(f'Cluster{gid_run}', fontdict=font_label)
    axs2.axis('equal')
    axs2.set_aspect('equal')
    fig2.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
    fig2.savefig(f'new/m_scatter/group{gid_run}.png')
    
    
    os.system(f'mkdir new/m_distribution')
    
    fig3, axs3 = plt.subplots(figsize=(9, 7))
    width = 2
    for bound in ['top', 'bottom', 'left', 'right']:
        axs3.spines[bound].set_linewidth(width)
    axs3.hist(v_rela, bins=30, alpha=0.5)
    axs3.tick_params(axis='both', direction='in', length=10, width=width, colors='black', labelsize=20)
    axs3.set_xlabel(r'$V_{relative}$ [km/s]', fontdict=font_label)
    axs3.set_ylabel(r'Number', fontdict=font_label)
    axs3.set_title(f'Cluster{gid_run}', fontdict=font_label)
    fig3.savefig(f'new/m_distribution/group{gid_run}.png')
    
    
    
os.system(f'mkdir new/m_info')
    
    
# Run
#for k in range(5):
'''
for k in range(len(gid_c)):
    run_cluster(gid_c[k], ra_c[k], dec_c[k], z_c[k])
'''
for k in range(182):
    group_id = k + 1
    ra_bcg, dec_bcg, z_bcg = get_bcg(group_id)
    run_cluster(group_id, ra_bcg, dec_bcg, z_bcg)

