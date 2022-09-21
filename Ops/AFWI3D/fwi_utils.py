import torch
import torch.nn.functional as F
from collections import OrderedDict
import json
import os
import numpy as np

def padding(cp, nz_orig, nx_orig, ny_orig, nz, nx, ny, nPml):
    tran_cp = cp.view(1, 1, nz_orig, nx_orig, ny_orig)
    tran_cp3 = F.pad(tran_cp, pad=(nPml, nPml, nPml, nPml, nPml, nPml), mode='replicate')
    cp_pad = tran_cp3.view(nz+2*nPml, nx+2*nPml, ny+2*nPml)
    return cp_pad


def paraGen(nz, nx, ny, dz, dx, dy, nSteps, dt, f0, nPml, para_fname, survey_fname, \
            data_dir_name, if_win=False, filter_para=None, if_src_update=False, \
            scratch_dir_name='', if_cross_misfit=False):
    # para = OrderedDict()
    para = {}
    para['nz'] = nz
    para['nx'] = nx
    para['ny'] = ny
    para['dz'] = dz
    para['dx'] = dx
    para['dy'] = dy
    para['nSteps'] = nSteps
    para['dt'] = dt
    para['f0'] = f0
    para['nPoints_pml'] = nPml

    if if_win != False:
        para['if_win'] = True

    if filter_para != None:
        para['filter'] = filter_para

    if if_src_update != False:
        para['if_src_update'] = True

    para['survey_fname'] = survey_fname
    para['data_dir_name'] = data_dir_name

    os.makedirs(data_dir_name, exist_ok=True)

    if if_cross_misfit != False:
        para['if_cross_misfit'] = True

    if (scratch_dir_name != ''):
        para['scratch_dir_name'] = scratch_dir_name
        os.makedirs(scratch_dir_name, exist_ok=True)

    with open(para_fname, 'w') as fp:
        json.dump(para, fp)


# all shots share the same number of receivers
def surveyGen(z_src, x_src, y_src, z_rec, x_rec, y_rec, survey_fname, Windows=None, \
              Weights=None, Src_Weights=None, Src_rxz=None, Rec_rxz=None):
    x_src = x_src.tolist()
    z_src = z_src.tolist()
    y_src = y_src.tolist()
    x_rec = x_rec.tolist()
    z_rec = z_rec.tolist()
    y_rec = y_rec.tolist()
    nsrc = len(x_src) * len(x_src[0])  # 震源的数量
    nrec = len(x_rec)  # 接收点的数量
    survey = {}
    survey['nShots'] = nsrc
    for i in range(0, nsrc):
        shot = {}
        shot['z_src'] = z_src[i % len(x_src)][i // len(x_src)]
        shot['x_src'] = x_src[i % len(x_src)][i // len(x_src)]
        shot['y_src'] = y_src[i % len(x_src)][i // len(x_src)]
        shot['nrec'] = nrec
        shot['z_rec'] = z_rec
        shot['x_rec'] = x_rec
        shot['y_rec'] = y_rec
        if Windows != None:
            shot['win_start'] = Windows['shot' + str(i)][:start]
            shot['win_end'] = Windows['shot' + str(i)][:end]

        if Weights != None:
            shot['weights'] = Weights['shot' + str(i)][:weights]

        if Src_Weights != None:
            shot['src_weight'] = Src_Weights[i]

        if Src_rxz != None:
            shot['src_rxz'] = Src_rxz[i]

        if Rec_rxz != None:
            Rec_rxz.tolist()
            shot['rec_rxz'] = Rec_rxz

        survey['shot' + str(i)] = shot

    with open(survey_fname, 'w') as fp:
        json.dump(survey, fp)

def sourceGene(f, nStep, delta_t):

  e = np.pi * np.pi * f * f
  t_delay = 1.2/f
  source = np.zeros((nStep))
  for it in range(0,nStep):
      source[it] = (1-2*e*(delta_t*(it)-t_delay)**2)*np.exp(-e*(delta_t*(it)-t_delay)**2)

  # return source

  for it in range(1,nStep):
      source[it] = source[it] + source[it-1]

  source = source * delta_t

  return source
  
