# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


# a = np.zeros((197, 2000), dtype=np.float32)
# a.tofile("a.bin")
traces = np.reshape(np.fromfile("/data/home/gkt/python_program/TorchFWI-master/TorchFWI-master/src/default/Data/Shot_x1.bin", dtype=np.float32), (197, 1678))
print(np.max(traces))
plt.imshow(traces[:, :].T, vmax=1e-4, vmin=-1e-4)
# plt.imshow(traces.T)
plt.colorbar()
plt.show()

# nPml = 32
# nPad = int(32 - np.mod((202+2*nPml), 32))  # 28
# VS = np.ascontiguousarray(np.reshape(np.load("../Mar_models/cs_pad.npy"), (160, -1), order='F'))[nPml:nPml+68,nPml:nPml+202]
# cs = sio.loadmat("default/Results/result25.mat", \
#   squeeze_me=True, struct_as_record=False)["cs"]
# plt.imshow(cs)
# plt.show()
# print(np.shape(traces))

