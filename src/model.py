# -*-coding:utf-8-*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import os

model_name = "models/marmousi2-model-true.mat"
smooth_model_name = "models/marmousi2-model-smooth.mat"
model = sio.loadmat(model_name)
smooth_model = sio.loadmat(smooth_model_name)

vp = np.load('models/vp_over_raw.npy')
vs = np.load('models/vs_over_raw.npy')
rho = np.load('models/rho_over_raw.npy')


sigma = 6
init_vp = vp.copy()
init_vp[:, 11:] = gaussian_filter(vp[:, 11:], [sigma,sigma], mode='reflect')

init_vs = vs.copy()
init_vs[:, 11:] = gaussian_filter(vs[:, 11:], [sigma,sigma], mode='reflect')

init_rho = rho.copy()
init_rho[:, 11:] = gaussian_filter(rho[:, 11:], [sigma,sigma], mode='reflect')

vs[:, :10] = 0
init_vs[:, :10] = 0
# init_rho[:, :11] = rho[:, :11]

nx=202
init_vp = np.mean(init_vp, axis=0, keepdims=True).repeat(nx, axis=0)
init_vs = np.mean(init_vs, axis=0, keepdims=True).repeat(nx, axis=0)

a = 14
init_vp[:, :a] = vp[:, :a]
init_vs[:, :a] = vs[:, :a]
init_rho[:, :a] = rho[:, :a]
vp[:, 50:] = 5500
# vs[:, 50:] = 3000
# rho[:, 40:50] = 2.3

# plt.plot(vp[10, :])
# plt.plot(vs[10, :])
# plt.plot(rho[10, :])
# plt.plot(init_vp[10, :])
# plt.plot(init_vs[10, :])
# plt.plot(init_rho[10, :])
np.save('models/vp_over.npy', vp)
np.save('models/vs_over.npy', vs)
np.save('models/rho_over.npy', rho)
np.save('models/init_vp_over.npy', init_vp)
np.save('models/init_vs_over.npy', init_vs)
np.save('models/init_rho_over.npy', init_rho)
plt.subplot(221)
plt.imshow(vp.T)
plt.subplot(222)
plt.imshow(vs.T)
plt.subplot(223)
plt.imshow(init_vp.T)
plt.subplot(224)
plt.imshow(init_vs.T)
print(np.max(vp))
print(np.min(vp))
print(np.max(vs))
print(np.min(vs))

plt.show()

sio.savemat(os.path.join('models', 'over_thrust_model.mat'),
         {"vp" : vp,
         "vs" : vs,
         "init_vp" : init_vp,
         "init_vs" : init_vs,
         "dx" : np.squeeze(model['dx']),
         "dy" : np.squeeze(model['dy']),
         "dt" : np.squeeze(model['dt']),
         "nx" : 202,
         "ny" : 68,
         "nt" : np.squeeze(model['nt']),
         "f0" : 3.0})




