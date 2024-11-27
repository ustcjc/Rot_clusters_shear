import emcee
import numpy as np
import matplotlib.pyplot as plt
import corner
import multiprocessing


#mode = 'bcg_major'
#mode = 'rotation'
mode = 'satellite_major'


# set some parameters (must be same as observation)

Mpc_step = 0.3
binsnum = 8 # one side
Mpc_range = Mpc_step * (binsnum-1/2)  # center +- Mpc_range
delta = 4 # pre-selection

xbins = np.linspace(-Mpc_range, Mpc_range, 2 * binsnum)
ybins = np.linspace(-Mpc_range, Mpc_range, 2 * binsnum)




from astropy.cosmology import Planck18 as cosmo
from astropy import constants as c
import astropy.units as u
def Sig_crit(zl, zs):
    fac = c.c**2 / (4 * np.pi * c.G)
    fac = fac.to(u.Msun * u.Mpc**(-2)*u.Mpc)
    Sig = fac * cosmo.angular_diameter_distance(zs) / (cosmo.angular_diameter_distance(zl) * cosmo.angular_diameter_distance_z1z2(zl, zs))
    return Sig




# define model

#factor = 1   # density(rho) --- convergence(kappa)
factor = 1 / Sig_crit(0.02, 0.1).value
def kappa_sph(rhos, rs, zeta):   # rhos, rs,  ---  Mvir, cvir,  (should, but not necessary)
    x = zeta / rs
    rho = rhos / (x * (1 + x)**2)
    return rho * factor

def NFW_model(rhos, rs, e,   x, y):
    # return kappa as a function of [x, y]
    if mode == 'satellite_major' or mode == 'bcg_major':
        theta = 0  # for major
    elif mode == 'rotation':
        theta = 90
    xp = x * np.cos(theta / 180 * np.pi) + y * np.sin(theta / 180 * np.pi)
    yp =  - x * np.sin(theta / 180 * np.pi) + y * np.cos(theta / 180 * np.pi)
    zeta = np.sqrt(xp**2 / (1 - e) + (1 - e) * yp**2)
    kappa = kappa_sph(rhos, rs, zeta)
    return kappa

def model(params, X, Y):
    rhos, rs, e, = params
    
    kappa = NFW_model(rhos, rs, e, X, Y)
    f = kappa
    
    # 初始化解 u
    u = np.zeros((len(ybins), len(xbins)))

    # 迭代求解
    dx = Mpc_step
    dy = Mpc_step
    tolerance = 1e-6
    max_iterations = 10000
    for it in range(max_iterations):
        u_old = u.copy()
        u[1:-1, 1:-1] = 0.25 * (u_old[1:-1, :-2] + u_old[1:-1, 2:] +
                                 u_old[:-2, 1:-1] + u_old[2:, 1:-1] +
                                 dx**2 * f[1:-1, 1:-1])

        # 边界条件
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]

        # 检查收敛
        if np.linalg.norm(u - u_old) < tolerance:
            break
    
    
    # 初始化导数数组
    u_xx = np.zeros_like(u)
    u_yy = np.zeros_like(u)
    u_xy = np.zeros_like(u)

    # 计算二阶偏导数
    for i in range(1, len(xbins)-1):
        for j in range(1, len(ybins)-1):
            u_xx[i, j] = (u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / dx**2
            u_yy[i, j] = (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / dy**2
            u_xy[i, j] = (u[i+1, j+1] - u[i+1, j-1] - u[i-1, j+1] + u[i-1, j-1]) / (4 * dx * dy)

    g1m = 1/2 * (u_xx - u_yy)
    g2m = - u_xy

    
    return [g1m, g2m]



# define likelihood and probability

def log_likelihood(params, X, Y, g1, g2, gerr):
    rhos, rs, e, = params
    if (rhos < 0) or (rs < 0) or (e < 0) or (e > 1):
        return -np.inf
    else:
        gamma = model(params, X, Y)
        g1m = gamma[0]
        g2m = gamma[1]
        chi2 = np.sum( ((g1 - g1m)/gerr)**2 + ((g2 - g2m)/gerr)**2 )
        return -0.5 * chi2

def log_prior(params):
    rhos, rs, e, = params
    if mode == 'satellite_major':
        if (0 < rhos < 7e14) & (0 < rs < 1) & (0 < e < 1):
            return 0.0
        else:
            return -np.inf
    elif mode == 'rotation':
        if (0 < rhos < 5e14) & (0 < rs < 0.8) & (0 < e < 1):
            return 0.0
        else:
            return -np.inf
    elif mode == 'bcg_major':
        if (0 < rhos < 5e14) & (0 < rs < 0.8) & (0 < e < 1):
            return 0.0
        else:
            return -np.inf
    
# probability
def log_probability(params, X, Y, g1, g2, gerr):
    return log_prior(params) + log_likelihood(params, X, Y, g1, g2, gerr)


# import observation gamma

sample = np.loadtxt(f'stack_{mode}.txt', unpack=True)
dxlist = sample[0]
dylist = sample[1]
g1list = sample[2]
g2list = sample[3]
gerrlist = sample[4]   # here gerr == sigma**2

X, Y = np.meshgrid(xbins, ybins)
g1 = np.zeros_like(X)
g2 = np.zeros_like(X)
gerr = np.zeros_like(X)

for i in range(len(X)):
    for j in range(len(X[0])):
        mask = (X[i][j] == dxlist) * (Y[i][j] == dylist)
        g1[i][j] = g1list[mask][0]
        g2[i][j] = g2list[mask][0]
        gerr[i][j] = np.sqrt(gerrlist[mask][0])



# set MCMC parameters and run

ndim = 3
nwalkers = 40
nsteps = 1000
skip_steps = 200

if mode == 'satellite_major':
    p1 = 2e14
    p2 = 0.3
    p3 = 0.4
    p0_min = [p1, p2, p3]
    p0_max = [p1+1e14, p2+0.2, p3+0.1]
elif mode == 'rotation':
    p1 = 1.5e14
    p2 = 0.3
    p3 = 0.2
    p0_min = [p1, p2, p3]
    p0_max = [p1+1e14, p2+0.2, p3+0.1]
elif mode == 'bcg_major':
    p1 = 2e14
    p2 = 0.3
    p3 = 0.2
    p0_min = [p1, p2, p3]
    p0_max = [p1+1e14, p2+0.2, p3+0.1]

initial = np.random.uniform(low=p0_min, high=p0_max, size=(nwalkers, ndim))

with multiprocessing.Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(X, Y, g1, g2, gerr), pool=pool)
    sampler.run_mcmc(initial, nsteps, progress=True)
    
samples = sampler.chain
flat_samples = samples[:, skip_steps:, :].reshape((-1, ndim))
'''
print(len(flat_samples.T))
best_fit_params = []
for k in range(len(flat_samples.T)):
    hist, bin_edges = np.histogram(flat_samples.T[k], bins=30)
    peak_index = np.argmax(hist)
    best_fit_params.append((bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2)
best_fit_params = np.array(best_fit_params)
'''
best_fit_params = np.median(flat_samples, axis=0)
stds = np.std(flat_samples, axis=0)










fig1, axes1 = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
print(samples.shape)
labels = [r'$\rho_s$ $\mathrm{[M_{\odot}/Mpc^2]}$', r'$r_s$ $\mathrm{[Mpc]}$', 'e']
for i in range(3):
    ax = axes1[i]
    ax.plot(samples[:, :, i], 'k', alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes1[-1].set_xlabel('step number')
fig1.savefig(f'mcmc_result/steps-2dmap-{mode}.png')




f = open(f'mcmc_result/info-2dmap-{mode}.txt', 'w+')
print(f'rhos = {best_fit_params[0]} | {stds[0]}', file=f)
print(f'rs = {best_fit_params[1]} | {stds[1]}', file=f)
print(f'e = {best_fit_params[2]} | {stds[2]}', file=f)
f.close()
values = best_fit_params






flat_samples = sampler.get_chain(discard=100, flat=True)[:,:3]
print(flat_samples.shape)
fig3 = corner.corner(
    flat_samples, labels=labels, truths=[values[0], values[1], values[2]]
)
fig3.savefig(f'mcmc_result/corner-2dmap-{mode}.png')

