# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


# a = np.zeros((197, 2000), dtype=np.float32)
# a.tofile("a.bin")
# nPml = 32
# nPad = int(32 - np.mod((202+2*nPml), 32))  # 28
# VS = np.ascontiguousarray(np.reshape(np.load("../Mar_models/cs_pad.npy"), (160, -1), order='F'))[nPml:nPml+68,nPml:nPml+202]

cs = sio.loadmat(
    "default/Results/cp70.mat",
                squeeze_me=True, struct_as_record=False)["cs"]

plt.imshow(cs)

plt.colorbar()

plt.show()
