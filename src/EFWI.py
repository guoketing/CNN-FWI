import torch
import numpy as np
import scipy.io as sio
import scipy
import sys
import os
import obspy
sys.path.append("../Ops/AFWI")
from FWI_ops import *
import matplotlib.pyplot as plt
import fwi_utils as ft
import argparse
from scipy import optimize
from obj_wrapper import PyTorchObjective
import time
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# get parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument('--generate_data', action='store_true')
parser.add_argument('--exp_name', type=str, default='default')
parser.add_argument('--nIter', type=int, default=10000)
parser.add_argument('--ngpu', type=int, default=1)
args = vars(parser.parse_args())
generate_data = args['generate_data']
exp_name = args['exp_name']
nIter = args['nIter']
ngpu = args['ngpu']

def extract_data(meta):
    data = []
    for trace in meta:
        data.append(trace.data)
    return np.array(data)

# ========== parameters ============
oz = 0.0  # original depth
ox = 0.0
dz_orig = 20
dx_orig = 20
nz_orig = 250  # original scale
nx_orig = 600  # original scale

dz = dz_orig / 1.0
dx = dx_orig / 1.0
nz = round((dz_orig * nz_orig) / dz)
nx = round((dx_orig * nx_orig) / dx)

dt = 0.008
nSteps = 1000

nPml = 20
nPad = 0
nz_pad = nz + 2 * nPml + nPad
nx_pad = nx + 2 * nPml

Mask = np.zeros((nz_pad, nx_pad))
# Mask[nPml:nPml+nz, nPml:nPml+nx] = 1.0
Mask[nPml:-nPml, nPml:-nPml] = 1.0
th_mask = torch.tensor(Mask, dtype=torch.float32)

# Mask = np.zeros((nz_pad, nx_pad))
# Mask[nPml:nPml+nz, nPml:nPml+nx] = 1.0
# Mask[nPml:nPml + 12,:] = 0.0
# th_mask = torch.tensor(Mask, dtype=torch.float32)


f0_vec = [14]
if_src_update = False
if_win = False

ind_src_x = np.arange(0, nx, 15).astype(int)
ind_src_z = 0 * np.ones(ind_src_x.shape[0]).astype(int)
n_src = ind_src_x.shape[0]

# ind_src_x = np.array([200])
# ind_src_z = np.array([0])
# n_src = 1

ind_rec_x = np.arange(3, nx).astype(int)
ind_rec_z = 1 * np.ones(ind_rec_x.shape[0]).astype(int)

para_fname = './' + exp_name + '/para_file_mar.json'
survey_fname = './' + exp_name + '/survey_file_mar.json'
data_dir_name = './' + exp_name + '/Data_mar'

wavelet = obspy.read("models/SEAM_wavelet-g_8ms.sgy", format='segy')

wavelet = extract_data(wavelet)[0]

# t = np.arange(len(wavelet))*dt
# f_vec = np.fft.fft(wavelet)
# f = np.fft.fftfreq(len(wavelet), dt)
# plt.plot(f[:len(f)//2], np.abs(f_vec)[:len(f)//2]/1e6, label="Original Ricker")
# plt.show()
# print(np.shape(wavelet))

# wavelet = sio.loadmat("../Mar_models/sourceF_4p5_2_high.mat", \
#                   squeeze_me=True, struct_as_record=False)["sourceF"]

if (len(wavelet) < nSteps):
    sou = np.zeros(nSteps)
    sou[:len(wavelet)] = wavelet
else:
    sou = wavelet[:nSteps]

cp_true_pad = np.load("models/vp.npy").T
assert np.shape(cp_true_pad) == (nz, nx)

timestep = Timestep(dt, dx, dz, np.max(cp_true_pad))

print(timestep.step_ratio)
print(timestep.inner_dt)


source_resampled = scipy.signal.resample(sou, nSteps * timestep.step_ratio)


# plt.plot(source_resampled)
# print(source_resampled)
# plt.show()
#
#
#
# raise RecursionError("e")
# print(source_resampled)


th_Stf = torch.tensor(source_resampled, dtype=torch.float32, requires_grad=False).repeat(n_src, 1)

ft.paraGen(nz_pad, nx_pad, dz, dx, nSteps * timestep.step_ratio, timestep.inner_dt,
        f0_vec[0], nPml, nPad, para_fname, survey_fname, data_dir_name)
ft.surveyGen(ind_src_z, ind_src_x, ind_rec_z, ind_rec_x, survey_fname)

Shot_ids = torch.tensor(np.arange(0, n_src), dtype=torch.int32)


opt = {}
opt['nz'] = nz
opt['nx'] = nx
opt['nz_orig'] = nz_orig
opt['nx_orig'] = nx_orig
opt['nPml'] = nPml
opt['nPad'] = nPad
opt['para_fname'] = para_fname
print(opt)

if (generate_data == True):
    print(f'cp_true_pad shape = {cp_true_pad.shape}')
    # den_true_pad = np.ascontiguousarray(np.reshape(np.load("models/rho_over.npy"), (nx, nz), order='F')).T
    th_cp_pad = torch.tensor(cp_true_pad, dtype=torch.float32, requires_grad=False)

    fwi_obscalc = FWI_obscalc(th_cp_pad, th_Stf, opt, para_fname)
    fwi_obscalc(Shot_ids, ngpu=ngpu)
    sys.exit('End of Data Generation')

########## Inversion ###########

cp_init_pad = np.load("models/init_vp.npy").T

# den_init_pad =  np.ascontiguousarray(np.reshape(np.load("models/init_rho_over.npy"), (nx, nz), order='F')).T

plt.imshow(cp_init_pad, cmap='RdBu_r')
plt.colorbar()
plt.savefig("cp.png")
vp_max = np.max(cp_init_pad)

vp_nor = cp_init_pad / vp_max

th_cp_inv = torch.tensor(vp_nor, dtype=torch.float32, requires_grad=True)

# Vp_bounds = [1500.0 * np.ones((nz, nx)), 5500.0 * np.ones((nz, nx))]
Vp_bounds = None

fwi = FWI(th_cp_inv, th_Stf, opt, Mask=th_mask, Vp_bounds=Vp_bounds, vp_max=vp_max)

compLoss = lambda : fwi(Shot_ids, ngpu=ngpu)
obj = PyTorchObjective(fwi, compLoss)


__iter = 0
result_dir_name = './' + exp_name + '/Results'
def save_prog(x):
  global __iter
  os.makedirs(result_dir_name, exist_ok=True)
  with open(result_dir_name + '/loss.txt', 'a') as text_file:
    text_file.write("%d %s\n" % (__iter, obj.f))
  sio.savemat(result_dir_name + '/cp' + str(__iter) + '.mat',
    {'cp':fwi.Vp.cpu().detach().numpy() * vp_max})
  sio.savemat(result_dir_name + '/grad_cp' + str(__iter) +
    '.mat', {'grad_cp':fwi.Vp.grad.cpu().detach().numpy()})
  __iter = __iter + 1

maxiter = nIter
optimize.minimize(obj.fun, obj.x0, method='L-BFGS-B', jac=obj.jac, bounds=obj.bounds,
  tol=None, callback=save_prog, options={'disp': True, 'iprint': 101,
  'gtol': 1e-012, 'maxiter': maxiter, 'ftol': 1e-12, 'maxcor': 30, 'maxfun': 15000})
