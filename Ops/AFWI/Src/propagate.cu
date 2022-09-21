#include "utilities.h"


__global__ void propagate(float *d_un, float *d_uc, float *d_up,
	float *d_phiz, float *d_phix, float *d_Vp, float *d_sigma_z, float *d_sigma_x, \
	int nz, int nx, float dt, float dz, float dx, int nPml, int nPad, bool isFor){

  int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;

  int id = gidx*nz+gidz;

  float lap = 0.0;
  float du_dz = 0.0;
  float dphiz_dz = 0.0;
  float du_dx = 0.0;
  float dphix_dx = 0.0;

  float vp = d_Vp[id];
  int n = 4;
  if (isFor) {
        if (gidz>=n && gidz<=nz-n-1 && gidx>=n && gidx<=nx-n-1) {

		    lap = -5.0/2.0 * d_uc[id] / (dx * dx) - 5.0/2.0 * d_uc[id] / (dz * dz) +
                 (16.0 * (d_uc[id + 1] + d_uc[id - 1]) - (d_uc[id + 2] + d_uc[id - 2])) / (12.0 * dz * dz) +
                 (16.0 * (d_uc[id + nz] + d_uc[id - nz]) - (d_uc[id + 2 * nz] + d_uc[id - 2 * nz])) / (12.0 * dx * dx);

		    if (gidz >= (nPml) && gidz < (nz- nPml) && gidx >= (nPml) && gidx < (nx - nPml)) {
                d_un[id] = vp * vp * dt * dt * lap + 2 * d_uc[id] - d_up[id];
            } else  {
            du_dz = (8.0*(d_uc[id+1]-d_uc[id-1])-(d_uc[id+2]-d_uc[id-2]))/(dz*12.0);
            dphiz_dz = (8.0*(d_phiz[id+1]-d_phiz[id-1])-(d_phiz[id+2]-d_phiz[id-2]))/(dz*12.0);

    		du_dx = (8*(d_uc[id+nz]-d_uc[id-nz])-(d_uc[id+2*nz]-d_uc[id-2*nz]))/(dx*12.0);
            dphix_dx = (8*(d_phix[id+nz]-d_phix[id-nz])-(d_phix[id+2*nz]-d_phix[id-2*nz]))/(dx*12.0);

            d_un[id] =
                1.0 / (1.0 + dt * (d_sigma_z[gidz] + d_sigma_x[gidx]) / 2.0) *
                (vp * vp * dt * dt * (lap + dphiz_dz + dphix_dx) +
                 dt * (d_sigma_z[gidz] + d_sigma_x[gidx]) * d_up[id] / 2.0 +
                (2.0 * d_uc[id] - d_up[id]) - dt * dt * d_sigma_z[gidz] * d_sigma_x[gidx] * d_uc[id]);

             
              d_phiz[id] = d_phiz[id] - dt * (d_sigma_z[gidz] * d_phiz[id] +
                                     (d_sigma_z[gidz] - d_sigma_x[gidx]) * du_dz);
              d_phix[id] = d_phix[id] - dt * (d_sigma_x[gidx] * d_phix[id] +
                                    (d_sigma_x[gidx] - d_sigma_z[gidz]) * du_dx);
              
			 }
		}
	}

	else {
        if (gidz >= (nPml) && gidz < (nz- nPml) && gidx >= (nPml) && gidx < (nx - nPml)) {
		    lap = -5.0/2.0 * d_uc[id] / (dx * dx) - 5.0/2.0 * d_uc[id] / (dz * dz) +
                 (16.0 * (d_uc[id + 1] + d_uc[id - 1]) - (d_uc[id + 2] + d_uc[id - 2])) / (12.0 * dz * dz) +
                 (16.0 * (d_uc[id + nz] + d_uc[id - nz]) - (d_uc[id + 2 * nz] + d_uc[id - 2 * nz])) / (12.0 * dx * dx);
            d_up[id] = vp * vp * dt * dt * lap + 2 * d_uc[id] - d_un[id];

		}
	}
}
