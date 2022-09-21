# -*-coding:utf-8-*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import os
import obspy


def extract_data(meta):
    data = []
    for trace in meta:
        data.append(trace.data)
    return np.array(data)

model_name = "models/SEAM_Vp_Elastic_N23900.sgy"
vp = obspy.read(model_name)
vp = extract_data(vp)[:, ::2][:600, :250]
print(np.shape(vp))
sigma = 10
init_vp = vp.copy()
init_vp[:, 11:] = gaussian_filter(vp[:, 11:], [sigma,sigma], mode='reflect')


# nx = 876
# init_vp = np.mean(init_vp, axis=0, keepdims=True).repeat(nx, axis=0)

# a = 20
# init_vp[:, :a] = vp[:, :a]

# vs[:, 50:] = 3000
# rho[:, 40:50] = 2.3

# plt.plot(vp[10, :])
# plt.plot(vs[10, :])
# plt.plot(rho[10, :])
# plt.plot(init_vp[10, :])
# plt.plot(init_vs[10, :])
# plt.plot(init_rho[10, :])
np.save('models/vp.npy', vp)
np.save('models/init_vp.npy', init_vp)

plt.subplot(211)
plt.imshow(vp.T)

plt.subplot(212)
plt.imshow(init_vp.T, vmax=np.max(vp), vmin=np.min(vp))

print(np.max(vp))
print(np.min(vp))

plt.show()

# sio.savemat(os.path.join('models', 'over_thrust_model.mat'),
#          {"vp" : vp,
#          "vs" : vs,
#          "init_vp" : init_vp,
#          "init_vs" : init_vs,
#          "dx" : np.squeeze(model['dx']),
#          "dy" : np.squeeze(model['dy']),
#          "dt" : np.squeeze(model['dt']),
#          "nx" : 202,
#          "ny" : 68,
#          "nt" : np.squeeze(model['nt']),
#          "f0" : 3.0})




