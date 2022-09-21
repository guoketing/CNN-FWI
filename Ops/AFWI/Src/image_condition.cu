#include "utilities.h"


__global__ void image_condition(float *d_uc, float *d_uc_adj,
	int nz, int nx, float dz, float dx, int nPml, float *d_Vp, float *d_VpGrad){

  int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;

  int id = gidx*nz+gidz;

  float lap = 0.0;

  if (gidz >= (nPml + 2) && gidz < (nz- nPml - 2) && gidx >= (nPml + 2) && gidx < (nx - nPml - 2)) {
    lap = -5.0/2.0 * d_uc[id] / (dx * dx) - 5.0/2.0 * d_uc[id] / (dz * dz) +
                 (16.0 * (d_uc[id + 1] + d_uc[id - 1]) - (d_uc[id + 2] + d_uc[id - 2])) / (12.0 * dz * dz) +
                 (16.0 * (d_uc[id + nz] + d_uc[id - nz]) - (d_uc[id + 2 * nz] + d_uc[id - 2 * nz])) / (12.0 * dx * dx);

    d_VpGrad[id] -= d_uc_adj[id]  * lap  * 2 / d_Vp[id];

  }
}
