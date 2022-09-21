#include "utilities.h"


__global__ void propagate(float *d_un, float *d_uc, float *d_up,
	float *d_phiz, float *d_phix, float *d_phiy, float *d_psci, float *d_Vp,
	float *d_sigma_z, float *d_sigma_x, float *d_sigma_y,
	int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml, bool isFor){

  int gidz = blockIdx.x*blockDim.x + threadIdx.x;
  int gidx = blockIdx.y*blockDim.y + threadIdx.y;
  int gidy = blockIdx.z*blockDim.z + threadIdx.z;

  int id = gidy*(nz*nx)+gidx*nz+gidz;

  float lap = 0.0;
  float du_dz = 0.0;
  float dphiz_dz = 0.0;
  float dpsci_dz = 0.0;
  float du_dx = 0.0;
  float dphix_dx = 0.0;
  float dpsci_dx = 0.0;
  float du_dy = 0.0;
  float dphiy_dy = 0.0;
  float dpsci_dy = 0.0;

  float vp = d_Vp[id];
  int n = 4;
  if (isFor) {
        if (gidz>=n && gidz<=nz-n-1 && gidx>=n && gidx<=nx-n-1 && gidy>=n && gidy<=ny-n-1) {

		    lap = -5.0/2.0 * d_uc[id] / (dx * dx) - 5.0/2.0 * d_uc[id] / (dz * dz) - 5.0/2.0 * d_uc[id] / (dy * dy) +
                 (16.0 * (d_uc[id + 1] + d_uc[id - 1]) - (d_uc[id + 2] + d_uc[id - 2])) / (12.0 * dz * dz) +
                 (16.0 * (d_uc[id + nz] + d_uc[id - nz]) - (d_uc[id + 2 * nz] + d_uc[id - 2 * nz])) / (12.0 * dx * dx) +
                 (16.0 * (d_uc[id + (nz*nx)] + d_uc[id - (nz*nx)]) - (d_uc[id + 2 * (nz*nx)] + d_uc[id - 2 * (nz*nx)])) / (12.0 * dy * dy);

		    if (gidz >= (nPml) && gidz < (nz- nPml) && gidx >= (nPml) && gidx < (nx - nPml) && gidy >= (nPml) && gidy < (ny - nPml)) {
                d_un[id] = vp * vp * dt * dt * lap + 2 * d_uc[id] - d_up[id];
            } else  {
            du_dz = (8.0*(d_uc[id+1]-d_uc[id-1])-(d_uc[id+2]-d_uc[id-2]))/(dz*12.0);
            dphiz_dz = (8.0*(d_phiz[id+1]-d_phiz[id-1])-(d_phiz[id+2]-d_phiz[id-2]))/(dz*12.0);
            dpsci_dz = (8.0*(d_psci[id+1]-d_psci[id-1])-(d_psci[id+2]-d_psci[id-2]))/(dz*12.0);

    		du_dx = (8*(d_uc[id+nz]-d_uc[id-nz])-(d_uc[id+2*nz]-d_uc[id-2*nz]))/(dx*12.0);
            dphix_dx = (8*(d_phix[id+nz]-d_phix[id-nz])-(d_phix[id+2*nz]-d_phix[id-2*nz]))/(dx*12.0);
            dpsci_dx = (8*(d_psci[id+nz]-d_psci[id-nz])-(d_psci[id+2*nz]-d_psci[id-2*nz]))/(dx*12.0);

    		du_dy = (8*(d_uc[id+(nz*nx)]-d_uc[id-(nz*nx)])-(d_uc[id+2*(nz*nx)]-d_uc[id-2*(nz*nx)]))/(dy*12.0);
            dphiy_dy = (8*(d_phiy[id+(nz*nx)]-d_phiy[id-(nz*nx)])-(d_phiy[id+2*(nz*nx)]-d_phiy[id-2*(nz*nx)]))/(dy*12.0);
            dpsci_dy = (8*(d_psci[id+(nz*nx)]-d_psci[id-(nz*nx)])-(d_psci[id+2*(nz*nx)]-d_psci[id-2*(nz*nx)]))/(dy*12.0);



            d_un[id] =
                1.0 / (1.0 + dt * (d_sigma_z[gidz] + d_sigma_x[gidx] + d_sigma_y[gidy]) / 2.0) *
                (vp * vp * dt * dt * lap + dt * dt * (dphiz_dz + dphix_dx + dphiy_dy -
                d_sigma_z[gidz] * d_sigma_x[gidx] * d_sigma_y[gidy] * d_psci[id]) +
                 dt * (d_sigma_z[gidz] + d_sigma_x[gidx] + d_sigma_y[gidy]) * d_up[id] / 2.0 +
                (2.0 * d_uc[id] - d_up[id]) - dt * dt * (d_sigma_z[gidz] * d_sigma_x[gidx] + d_sigma_y[gidy] * d_sigma_x[gidx] +
                d_sigma_z[gidz] * d_sigma_y[gidy]) * d_uc[id]);

             
              d_phiz[id] = d_phiz[id] - dt * d_sigma_z[gidz] * d_phiz[id] +
                                     vp * vp * dt * (d_sigma_y[gidy] + d_sigma_x[gidx]) * du_dz +
                                     dt * d_sigma_y[gidy] * d_sigma_x[gidx] * dpsci_dz;

              d_phix[id] = d_phix[id] - dt * d_sigma_x[gidx] * d_phix[id] +
                                    vp * vp * dt * (d_sigma_z[gidz] + d_sigma_y[gidy]) * du_dx +
                                    dt * d_sigma_z[gidz] * d_sigma_y[gidy] * dpsci_dx;

              d_phiy[id] = d_phiy[id] - dt * d_sigma_y[gidy] * d_phiy[id] +
                                    vp * vp * dt * (d_sigma_z[gidz] + d_sigma_x[gidx]) * du_dy +
                                    dt * d_sigma_z[gidz] * d_sigma_x[gidx] * dpsci_dy;

              d_psci[id] += dt * d_uc[id];
              
			 }
		}
	}

	else {
        if (gidz >= (nPml) && gidz < (nz- nPml) && gidx >= (nPml) && gidx < (nx - nPml) && gidy >= (nPml) && gidy < (ny - nPml)) {
		   lap = -5.0/2.0 * d_uc[id] / (dx * dx) - 5.0/2.0 * d_uc[id] / (dz * dz) - 5.0/2.0 * d_uc[id] / (dy * dy) +
                 (16.0 * (d_uc[id + 1] + d_uc[id - 1]) - (d_uc[id + 2] + d_uc[id - 2])) / (12.0 * dz * dz) +
                 (16.0 * (d_uc[id + nz] + d_uc[id - nz]) - (d_uc[id + 2 * nz] + d_uc[id - 2 * nz])) / (12.0 * dx * dx) +
                 (16.0 * (d_uc[id + (nz*nx)] + d_uc[id - (nz*nx)]) - (d_uc[id + 2 * (nz*nx)] + d_uc[id - 2 * (nz*nx)])) / (12.0 * dy * dy);
            d_up[id] = vp * vp * dt * dt * lap + 2 * d_uc[id] - d_un[id];

		}
	}
}
