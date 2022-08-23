import torch
import numpy as np
import scipy.io as sio
import sys
import os

sys.path.append("../Ops/FWI")
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

model_name = "models/marmousi2-model-true.mat"
smooth_model_name = "models/marmousi2-model-smooth.mat"
model = sio.loadmat(model_name)
smooth_model = sio.loadmat(smooth_model_name)

# ========== parameters ============
oz = 0.0  # original depth
ox = 0.0
dz_orig = np.squeeze(model["dy"])
dx_orig = np.squeeze(model["dx"])
nz_orig = np.squeeze(model["ny"])  # original scale
nx_orig = np.squeeze(model["nx"])  # original scale

dz = dz_orig / 1.0
dx = dx_orig / 1.0
nz = round((dz_orig * nz_orig) / dz)
nx = round((dx_orig * nx_orig) / dx)

dt = 0.0025
nSteps = 1800

nPml = 32
nPad = 0
nz_pad = nz + 2 * nPml + nPad
nx_pad = nx + 2 * nPml

Mask = np.zeros((nz_pad, nx_pad))
# Mask[nPml:nPml+nz, nPml:nPml+nx] = 1.0
Mask[nPml + 14:, :] = 1.0
th_mask = torch.tensor(Mask, dtype=torch.float32)

# Mask = np.zeros((nz_pad, nx_pad))
# Mask[nPml:nPml+nz, nPml:nPml+nx] = 1.0
# Mask[nPml:nPml + 12,:] = 0.0
# th_mask = torch.tensor(Mask, dtype=torch.float32)


f0_vec = [4.5]
if_src_update = False
if_win = False

ind_src_x = np.arange(0, nx, 3).astype(int)
ind_src_z = 2 * np.ones(ind_src_x.shape[0]).astype(int)
n_src = ind_src_x.shape[0]

ind_rec_x = np.arange(3, nx).astype(int)
ind_rec_z = 9 * np.ones(ind_rec_x.shape[0]).astype(int)

para_fname = './' + exp_name + '/para_file_mar.json'
survey_fname = './' + exp_name + '/survey_file_mar.json'
data_dir_name = './' + exp_name + '/Data_mar'
ft.paraGen(nz_pad, nx_pad, dz, dx, nSteps, dt, f0_vec[0], nPml, nPad, para_fname, survey_fname, data_dir_name)
ft.surveyGen(ind_src_z, ind_src_x, ind_rec_z, ind_rec_x, survey_fname)

Stf = sio.loadmat("../Mar_models/sourceF_4p5_2_high.mat", \
                  squeeze_me=True, struct_as_record=False)["sourceF"]
n = len(Stf)
src = np.zeros(nSteps)

if nSteps > n:
    src[:n] = Stf
else:
    src = Stf[:nSteps]

th_Stf = torch.tensor(src[:nSteps], dtype=torch.float32, \
                      requires_grad=False).repeat(n_src, 1)

Shot_ids = torch.tensor(np.arange(0, n_src), dtype=torch.int32)
# Shot_ids = torch.tensor([24], dtype=torch.int32)
# ==========

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
    cp_true_pad = np.ascontiguousarray(np.reshape(np.load("models/vp_over.npy"), (nx, nz), order='F')).T
    cs_true_pad = np.ascontiguousarray(np.reshape(np.load("models/vs_over.npy"), (nx, nz), order='F')).T
    print(f'cp_true_pad shape = {cp_true_pad.shape}')
    plt.imshow(cs_true_pad, cmap='RdBu_r')
    plt.colorbar()
    plt.savefig("cp.png")
    den_true_pad = 2.5 * np.ones((nx, nz))
    # den_true_pad = np.ascontiguousarray(np.reshape(np.load("models/rho_over.npy"), (nx, nz), order='F')).T
    th_cp_pad = torch.tensor(cp_true_pad, dtype=torch.float32, requires_grad=False)
    th_cs_pad = torch.tensor(cs_true_pad, dtype=torch.float32, requires_grad=False)
    th_den_pad = torch.tensor(den_true_pad, dtype=torch.float32, requires_grad=False)

    fwi_obscalc = FWI_obscalc(th_cp_pad, th_cs_pad, th_den_pad, th_Stf, opt, para_fname)
    fwi_obscalc(Shot_ids, ngpu=ngpu)
    sys.exit('End of Data Generation')

########## Inversion ###########

cp_init_pad = sio.loadmat(
    "results/result220.mat",
                squeeze_me=True, struct_as_record=False)["cp"]
#
# cp = cp_init_pad[50:, :]
# cp[cp>5300] = 5300
# cp_init_pad[50:, :] = cp
# plt.imshow(cp_init_pad)
# plt.show()
# raise  RecursionError("E")

# cp_init_pad = np.ascontiguousarray(np.reshape(np.load("models/init_vp_over.npy"), (nx, nz), order='F')).T
# cs_init_pad = np.ascontiguousarray(np.reshape(np.load("models/init_vs_over.npy"), (nx, nz), order='F')).T
cs_init_pad = sio.loadmat(
    "results/result220.mat",
                squeeze_me=True, struct_as_record=False)["cs"]

# cs = cs_init_pad[40:, :]
# cs[cs>3100] = 3100
# cs_init_pad[40:, :] = cs

# plt.imshow(cs_init_pad)
# plt.show()
# raise RecursionError("E")
den_init_pad =  2.5*np.ones((nx, nz))

# den_init_pad =  np.ascontiguousarray(np.reshape(np.load("models/init_rho_over.npy"), (nx, nz), order='F')).T
plt.imshow(den_init_pad, cmap='RdBu_r')
plt.colorbar()
plt.savefig("cp.png")
vp_max = np.max(cp_init_pad)
vs_max = np.max(cs_init_pad)
rho_max = np.max(den_init_pad)

vp_nor = cp_init_pad / vp_max
vs_nor = cs_init_pad / vs_max
# rho_nor = den_init_pad / rho_max

th_cp_inv = torch.tensor(vp_nor, dtype=torch.float32, requires_grad=False)
th_cs_inv = torch.tensor(vs_nor, dtype=torch.float32, requires_grad=False)
th_den_inv = torch.tensor(den_init_pad, dtype=torch.float32, requires_grad=False)

# Vp_bounds = [1500.0 * np.ones((nz, nx)), 5500.0 * np.ones((nz, nx))]
Vp_bounds = None

seed = 111

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True

seed_torch(seed)

Shot_ids = torch.tensor(np.arange(0, n_src), dtype=torch.int32)
z = np.random.rand(1, 8)
z = torch.tensor(z, dtype=torch.float32, requires_grad=False)
x = np.random.rand(1, 8)
x = torch.tensor(x, dtype=torch.float32, requires_grad=False)

fwi = NN_FWI(th_cp_inv, th_cs_inv, th_den_inv, th_Stf, opt, Mask=th_mask, Vp_max=vp_max, \
        Vs_max=vs_max, Den_max=rho_max, h0=5, w0=13)

num_batches = 34
figure_dir_name = "figure"
result_dir_name = "results"
criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(fwi.parameters(), 3e-4)

for epoch in range(10000):
    # start = time.time()
    # optimizer.zero_grad()
    # comLoss = fwi(z, x, 1.5, -1.5, Shot_ids, ngpu=ngpu)
    # zero = torch.tensor(np.zeros(1), dtype=torch.float32)
    # loss = criterion(zero, comLoss)
    # epoch_loss = loss.item()
    # loss.backward()
    # optimizer.step()
    # end = time.time()
    # # print("time :", end - start, "s")
    # print(epoch, epoch_loss)


    epoch_loss = 0.0
    start = time.time()
    for batch in range(num_batches):
        optimizer.zero_grad()
        batch_shots = torch.tensor(np.arange(0+batch, n_src, num_batches), dtype=torch.int32)
        comLoss = fwi(z, x, 1.5, -1.5, batch_shots, ngpu=ngpu)
        zero = torch.tensor(np.zeros(1), dtype=torch.float32)
        loss = criterion(zero, comLoss)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    end = time.time()
    # print("time :", end - start, "s")
    print(epoch, "epoch_loss = " + '{:g}'.format(epoch_loss))

    # if epoch == 50:
    #   optimizer = torch.optim.Adam(fwi.parameters(), 7e-4)
    # flag = 0
    # if epoch_loss <= 1e-2 and flag==0:
    #   flag = 1
    #   optimizer = torch.optim.Adam(fwi.parameters(), 7e-4)

    # if epoch_loss <= 1e-3 and flag == 1:
    #   flag = 2
    #   optimizer = torch.optim.Adam(fwi.parameters(), 1e-4)

    # if epoch == 200:
    #   optimizer = torch.optim.Adam(fwi.parameters(), 1e-5)

    if epoch % 2 == 0:
      plt.close("all")

      plt.imshow(fwi.Vp_mask_pad.cpu().detach().numpy()[nPml:-nPml-nPad, nPml:-nPml])
      plt.title("Vp")
      plt.colorbar()
      plt.savefig(figure_dir_name + "/vp/" + str(epoch) + '.png')
      plt.close("all")
      plt.imshow(fwi.Vs_mask_pad.cpu().detach().numpy()[nPml:-nPml-nPad, nPml:-nPml])
      plt.title("Vs")
      plt.colorbar()
      plt.savefig(figure_dir_name + "/vs/" + str(epoch) + '.png')
      # plt.close("all")
      # plt.imshow(fwi.Den_mask_pad.cpu().detach().numpy()[nPml:-nPml-nPad, nPml:-nPml])
      # plt.title("Den")
      # plt.colorbar()
      # plt.savefig(figure_dir_name + "/Den/" + str(epoch) + '.png')
    # with open(result_dir_name + '/loss.txt', 'a') as text_file:
    #   text_file.write("%d %s\n" % (epoch, epoch_loss))
      sio.savemat(result_dir_name + '/result' + str(epoch) + '.mat', \
          {'cp':fwi.Vp_mask_pad.cpu().detach().numpy()[nPml:-nPml-nPad, nPml:-nPml],
           'cs':fwi.Vs_mask_pad.cpu().detach().numpy()[nPml:-nPml-nPad, nPml:-nPml]})