import torch
import torch.nn as nn
import numpy as np 
from torch.utils.cpp_extension import load
# import matplotlib.pyplot as plt
import os
from scipy import optimize
import fwi_utils as ft
from collections import OrderedDict

abs_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(abs_path, 'Src')
os.makedirs(path+'/build/', exist_ok=True)

def load_fwi(path):
    fwi = load(name="FWI", sources=[path+'/Torch_Fwi.cpp', path+'/Parameter.cpp', path+'/libCUFD.cu', path+'/propagate.cu', path+'/image_condition.cu',
                                    path+'/Model.cu', path+'/Cpml.cu', path+'/utilities.cu', path+'/Src_Rec.cu',
                                    path+'/Boundary.cu'],
               extra_cflags=[
                   '-O3 -fopenmp -lpthread'
               ],
               extra_include_paths=['/usr/local/cuda/include', path + '/rapidjson'],
               extra_ldflags=['-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft'],
               build_directory=path + '/build/',
               verbose=True)
    return fwi

fwi_ops = load_fwi(path)

# class FWIFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, Lambda, Mu, Den, Stf, gpu_id, Shot_ids, para_fname):
#         misfit, = fwi_ops.forward(Lambda, Mu, Den, Stf, gpu_id, Shot_ids, para_fname)
#         variables = [Lambda, Mu, Den, Stf]
#         ctx.save_for_backward(*variables)
#         ctx.gpu_id = gpu_id
#         ctx.Shot_ids = Shot_ids
#         ctx.para_fname = para_fname
#         return misfit

#     @staticmethod
#     def backward(ctx, grad_misfit):
#         outputs = fwi_ops.backward(*ctx.saved_variables, ctx.gpu_id, ctx.Shot_ids, ctx.para_fname)
#         grad_Lambda, grad_Mu, grad_Den, grad_stf = outputs
#         return grad_Lambda, grad_Mu, grad_Den, grad_stf, None, None, None

class FWIFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Vp, Stf, ngpu, Shot_ids, para_fname):
        outputs = fwi_ops.backward(Vp, Stf, ngpu, Shot_ids, para_fname)
        ctx.outputs = outputs[1:]
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_misfit):
        grad_Vp, grad_stf = ctx.outputs
        return grad_Vp, grad_stf, None, None, None

class FWI(torch.nn.Module):
    def __init__(self, Vp, Stf, opt, Mask=None, Vp_bounds=None, vp_max=None):
        super(FWI, self).__init__()

        self.nz = opt['nz']
        self.nx = opt['nx']
        self.ny = opt['ny']
        self.nz_orig = opt['nz_orig']
        self.nx_orig = opt['nx_orig']
        self.ny_orig = opt['ny_orig']
        self.nPml = opt['nPml']

        self.vp_max = vp_max

        self.Bounds = {}
        Vp_pad= ft.padding(Vp, self.nz_orig, self.nx_orig, self.ny_orig, self.nz, self.nx, self.ny, self.nPml)
        Vp_ref = Vp_pad.clone().detach()
        self.register_buffer('Vp_ref', Vp_ref)
        if Vp.requires_grad:
            self.Vp = nn.Parameter(Vp)
            if Vp_bounds != None:
                self.Bounds['Vp'] = Vp_bounds
        else:
            self.Vp = Vp

        if Mask == None:
            self.Mask = torch.ones((self.nz+2*self.nPml, self.nx+2*self.nPml, self.ny+2*self.nPml), dtype=torch.float32)
        else:
            self.Mask = Mask

        self.Stf = Stf
        self.para_fname = opt['para_fname']

    def forward(self, Shot_ids, ngpu=1):
        Vp_pad= ft.padding(self.Vp,
            self.nz_orig, self.nx_orig, self.ny_orig, self.nz, self.nx, self.ny, self.nPml)
        Vp_mask_pad = self.Mask * Vp_pad + (1.0 - self.Mask) * self.Vp_ref
        Vp_mask_pad = Vp_mask_pad * self.vp_max

        return FWIFunction.apply(Vp_mask_pad, self.Stf, ngpu, Shot_ids, self.para_fname)

# class NN_FWI(torch.nn.Module):
#     def __init__(self, Vp, Vs, Den, Stf, opt, Mask=None, Vp_max=None, Vs_max=None, Den_max=None, h0=5, w0=13):
#         super(NN_FWI, self).__init__()
#
#         self.nz = opt['nz']
#         self.nx = opt['nx']
#         self.ny = opt['ny']
#         self.nz_orig = opt['nz_orig']
#         self.nx_orig = opt['nx_orig']
#         self.ny_orig = opt['ny_orig']
#         self.nPml = opt['nPml']
#         self.vp_max = Vp_max
#         self.vs_max = Vs_max
#         self.rho_max = Den_max
#
#         self.Bounds = {}
#         Vp_pad, Vs_pad, Den_pad = ft.padding(Vp, Vs, Den, self.nz_orig, self.nx_orig, self.ny_orig, self.nz, self.nx, self.nPml)
#         Vp_ref = Vp_pad.clone().detach()
#         self.Vp_mask_pad = Vp_pad.clone()
#         self.Vs_mask_pad = Vs_pad.clone()
#         self.Den_mask_pad = Den_pad.clone()
#         Vs_ref = Vs_pad.clone().detach()
#         Den_ref = Den_pad.clone().detach()
#         self.register_buffer('Vp_ref', Vp_ref)
#         self.register_buffer('Vs_ref', Vs_ref)
#         self.register_buffer('Den_ref', Den_ref)
#         self.act = nn.Tanh()
#         self.act2 = nn.LeakyReLU(0.1)
#         self.h0=h0
#         self.w0=w0
#
#         self.head_vp = nn.Linear(in_features=8, out_features=h0*w0*8)
#         self.body_vp = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=8, out_channels=128, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
#                              self.act2)
#         self.tail_vp = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(2, 2), stride=(1, 1), padding=(2, 2), bias=False))
#
#         self.head_vs = nn.Linear(in_features=8, out_features=h0*w0*8)
#         self.body_vs = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=8, out_channels=128, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
#                              self.act2)
#         self.tail_vs = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(2, 2), stride=(1, 1), padding=(2, 2), bias=False))
#
#         self.head_rho = nn.Linear(in_features=8, out_features=h0*w0*8)
#         self.body_rho = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=8, out_channels=128, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
#                              self.act2)
#         self.tail_rho = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Upsample(scale_factor=2, mode='bilinear'),
#                              nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
#                              self.act2,
#                              nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(2, 2), stride=(1, 1), padding=(2, 2), bias=False))
#
#
#         # if Vp.requires_grad:
#             # self.Vp = nn.Parameter(Vp)
#         # else:
#         self.Vp = Vp
#         self.Vs = Vs
#         self.Den = Den
#
#         if Mask == None:
#             self.Mask = torch.ones((self.nz+2*self.nPml, self.nx+2*self.nPml), dtype=torch.float32)
#         else:
#             self.Mask = Mask
#
#         self.Stf = Stf
#         self.para_fname = opt['para_fname']
#
#     def forward(self,z, x,  vmax, vmin, Shot_ids, ngpu=1):
#         Vp_pad, Vs_pad, Den_pad = ft.padding(self.Vp, self.Vs, self.Den, self.nz_orig, self.nx_orig, self.nz, self.nx, self.nPml)
#         self.Vp_mask_pad = self.Mask * Vp_pad + (1.0 - self.Mask) * self.Vp_ref
#         self.Vs_mask_pad = self.Mask * Vs_pad + (1.0 - self.Mask) * self.Vs_ref
#         self.Den_mask_pad = self.Mask * Den_pad + (1.0 - self.Mask) * self.Den_ref
#
#         delta_vp = self.head_vp(z)
#         delta_vp = torch.reshape(delta_vp, (-1, 8, self.h0, self.w0))
#         delta_vp = self.act(delta_vp)
#         delta_vp = self.body_vp(delta_vp)
#         delta_vp = self.tail_vp(delta_vp)
#         delta_vp = ((vmax-vmin) * self.act(delta_vp) + (vmax + vmin))/2.0
#         delta_vp = torch.squeeze(delta_vp)*0.65
#
#         delta_vs = self.head_vs(z)
#         delta_vs = torch.reshape(delta_vs, (-1, 8, self.h0, self.w0))
#         delta_vs = self.act(delta_vs)
#         delta_vs = self.body_vs(delta_vs)
#         delta_vs = self.tail_vs(delta_vs)
#         delta_vs = ((vmax-vmin) * self.act(delta_vs) + (vmax + vmin))/2.0
#         delta_vs = torch.squeeze(delta_vs)*0.4
#
#         # delta_rho = self.head_rho(z)
#         # delta_rho = torch.reshape(delta_rho, (-1, 8, self.h0, self.w0))
#         # delta_rho = self.act(delta_rho)
#         # delta_rho = self.body_rho(delta_rho)
#         # delta_rho = self.tail_rho(delta_rho)
#         # # delta_vs = ((vmax-vmin) * self.act(delta_vs) + (vmax + vmin))/2.0
#         # delta_rho = torch.squeeze(delta_rho) * 0.2
#
#
#         hv, wv = self.Vp_mask_pad.shape
#         hx, wx = delta_vp.shape
#         # vs_hx, vs_wx = delta_vp.shape
#         # print(np.shape(self.Vp_mask_pad))
#         # print(np.shape(delta_vp))
#         # print(np.shape(delta_vs))
#         # raise RecursionError("e")
#
#         if (hv > hx):
#             self.Vp_mask_pad[self.nPml:self.nPml+hx, (wv-wx)//2 +1:(wv-wx)//2 +1+wx] += delta_vp
#             self.Vs_mask_pad[self.nPml:self.nPml+hx, (wv-wx)//2 +1:(wv-wx)//2 +1+wx] += delta_vs
#             # self.Den_mask_pad[self.nPml:self.nPml+hx, (wv-wx)//2 +1:(wv-wx)//2 +1+wx] += delta_rho
#         else:
#             # self.Vp_mask_pad += delta_vp[(hx-hv)//2+1:(hx-hv)//2+1+hv, (wx-wv)//2 +1:(wx-wv)//2 +1+wv]
#             # self.Vs_mask_pad += delta_vs[(hx-hv)//2+1:(hx-hv)//2+1+hv, (wx-wv)//2 +1:(wx-wv)//2 +1+wv]
#
#             raise RecursionError("E")
#
#         self.Vp_mask_pad = self.Mask * self.Vp_mask_pad + (1.0 - self.Mask) * self.Vp_ref
#         self.Vs_mask_pad = self.Mask * self.Vs_mask_pad + (1.0 - self.Mask) * self.Vs_ref
#         # self.Den_mask_pad = self.Mask * self.Den_mask_pad + (1.0 - self.Mask) * self.Den_ref
#
#
#         # Den_mask_pad = Den_mask_pad / 1000
#         # plt.close("all")
#         # plt.imshow(self.Vp_mask_pad.detach().numpy(), cmap='RdBu_r')
#         # print(np.max(self.Vp_mask_pad.detach().numpy()))
#         # plt.colorbar()
#         # plt.savefig("cp.png")
#         self.Vp_mask_pad = self.Vp_mask_pad * self.vp_max
#         self.Vs_mask_pad = self.Vs_mask_pad * self.vs_max
#         # self.Den_mask_pad = self.Den_mask_pad * self.rho_max
#
#         self.Vp_mask_pad[self.Vp_mask_pad<1500] = 1500
#         self.Vp_mask_pad[self.Vp_mask_pad>5500] = 5500
#
#         self.Vs_mask_pad[self.Vs_mask_pad<0] = 0
#         self.Vs_mask_pad[self.Vs_mask_pad>3500] = 3500
#
#         # self.Den_mask_pad[self.Den_mask_pad<1000] = 1000
#         # self.Den_mask_pad[self.Den_mask_pad>3000] = 3000
#
#         Lambda = (self.Vp_mask_pad**2 - 2.0 * self.Vs_mask_pad**2) * self.Den_mask_pad
#         Mu = self.Vs_mask_pad**2 * self.Den_mask_pad
#         Den = self.Den_mask_pad
#         return FWIFunction.apply(Lambda, Mu, Den, self.Stf, ngpu, Shot_ids, self.para_fname)

class FWI_obscalc(torch.nn.Module):
    def __init__(self, Vp, Stf, opt, para_fname):
        super(FWI_obscalc, self).__init__()
        Vp = ft.padding(Vp, opt['nz_orig'], opt['nx_orig'], opt['ny_orig'], opt['nz'], opt['nx'], opt['ny'], opt['nPml'])
        self.Vp = Vp
        self.Stf = Stf
        self.para_fname = para_fname

    def forward(self, Shot_ids, ngpu=1):
        fwi_ops.obscalc(self.Vp, self.Stf, ngpu, Shot_ids, self.para_fname)

class Timestep(object):
    """The timestep used during wave propagation.

    The dt specified by the user during forward modeling is the time interval
    between source samples, and is also used for the time interval between
    output receiver samples and between the wavefields saved for
    backpropagation. The wave propagator may need a smaller time step size for
    numerical stability, which will be called inner_dt. It depends on the
    cell size and the maximum wave speed. This inner_dt will be chosen so
    that dt is a multiple of it. The factor will be called step_ratio.

    Args:
        model: Model object
        dt: A float specifying the time interval between source samples
    """

    def __init__(self, dt, dx, dz, max_vel):
        dh = np.min([dx, dz])
        max_dt = 0.6 * dh / np.sqrt(2.0) / max_vel
        self.step_ratio = int(np.ceil(dt / max_dt))
        self.inner_dt = dt / self.step_ratio



