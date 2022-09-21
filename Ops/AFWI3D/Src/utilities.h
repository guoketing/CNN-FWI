#ifndef UTILITIES_H__
#define UTILITIES_H__

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "cufft.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"

#define PI (3.141592653589793238462643383279502884197169)
// #define MEGA 1e6

#define TX 8
#define TY 8
#define TZ 8


__constant__ float coef[2];

#define CHECK(call)                                                      \
  {                                                                      \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                           \
    }                                                                    \
  }

void fileBinLoad(float *h_bin, int size, std::string fname);

void fileBinWrite(float *h_bin, int size, std::string fname);

void fileBinWriteDouble(double *h_bin, int size, std::string fname);

void initialArray(float *ip, int size, float value);

void initialArray(double *ip, int size, double value);

__global__ void intialArrayGPU(float *ip, int nz, int nx, int ny, float value);

__global__ void intial2dArrayGPU(float *ip, int nx, int ny, float value);

__global__ void assignArrayGPU(float *ip_in, float *ip_out, int nx, int ny);

void displayArray(std::string s, float *ip, int nx, int ny);

__global__ void moduliInit(float *d_Cp, float *d_Cs, float *d_Den,
                           float *d_Lambda, float *d_Mu, int nx, int ny);

__global__ void velInit(float *d_Lambda, float *d_Mu, float *d_Den, float *d_Cp,
                        float *d_Cs, int nx, int ny);

__global__ void aveMuInit(float *d_Mu, float *d_ave_Mu, int nx, int ny);

__global__ void aveBycInit(float *d_Den, float *d_ave_Byc_a, float *d_ave_Byc_b,
                           int nx, int ny);

__global__ void gpuMinus(float *d_out, float *d_in1, float *d_in2, int nx,
                         int ny);

__global__ void cuda_cal_objective(float *obj, float *err, int ng);

float cal_objective(float *array, int N);

float compCpAve(float *array, int N);

float compCourantNumber(float *h_Cp, int size, float dt, float dz, float dx);

void cpmlInit(float *sigma, int N, int nPml, float dh, float f0,
              float CpAve);

__global__ void image_condition(float *d_uc, float *d_uc_adj,
	int nz, int nx, int ny, float dz, float dx, float dy, int nPml, float *d_Vp, float *d_VpGrad);

__global__ void propagate(float *d_un, float *d_uc, float *d_up,
	float *d_phiz, float *d_phix, float *d_phiy, float *d_psci, float *d_Vp,
	float *d_sigma_z, float *d_sigma_x, float *d_sigma_y,
	int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml, bool isFor);

__global__ void src_rec_gauss_amp(float *gauss_amp, int nz, int nx, int ny);

__global__ void add_source(float *d_un, float amp, int nz, int nx,
                           bool isFor, int z_loc, int x_loc, int y_loc, float dt,
                           float *gauss_amp);

__global__ void recording(float *d_uc, int nz, int nx, float *d_data,
                          int iShot, int it, int nSteps, int nrec, int *d_z_rec,
                          int *d_x_rec, int *d_y_rec);

__global__ void res_injection(float *d_un_adj, int nz, int nx,
                              float *d_res, int it, float dt, int nSteps,
                              int nrec, int *d_z_rec, int *d_x_rec, int *d_y_rec);

__global__ void source_grad(float *d_szz_adj, float *d_sxx_adj, int nz,
                            float *d_StfGrad, int it, float dt, int z_src,
                            int x_src);

__global__ void minus_source(float *d_szz, float *d_sxx, float amp, int nz,
                             int z_loc, int x_loc, float dt, float *d_Cp);


__global__ void from_bnd(float *d_field, float *d_bnd, int nz, int nx, int ny,
                         int nzBnd, int nxBnd, int nyBnd, int len_Bnd_vec, int nLayerStore,
                         int indT, int nPml, int nSteps);

__global__ void cuda_normal_adjoint_source(
    int nt, int nrec, float *d_obs_normfact, float *d_cal_normfact,
    float *d_cross_normfact, float *d_data_obs, float *d_data, float *d_res,
    float *d_weights, float src_weight);


__global__ void to_bnd(float *d_field, float *d_bnd, int nz, int nx, int ny, int nzBnd,
                       int nxBnd, int nyBnd, int len_Bnd_vec, int nLayerStore, int indT,
                       int nPml, int nSteps);

__global__ void cuda_find_normfact(int nt, int nrec, float *data1, float *data2,
                                   float *normfact);


__global__ void exchange_wavefield(float *d_un, float *d_uc, float *d_up, int nz, int nx);


__global__ void cuda_bp_filter1d(int nSteps, float dt, int nrec,
                                 cufftComplex *d_data_F, float f0, float f1,
                                 float f2, float f3);

__global__ void cuda_filter1d(int nf, int nrec, cuFloatComplex *d_data_F,
                              cuFloatComplex *d_coef);

__global__ void cuda_normalize(int nz, int nx, float *data, float factor);


__global__ void cuda_window(int nt, int nrec, float dt, float *d_win_start,
                            float *d_win_end, float *d_weights,
                            float src_weight, float ratio, float *data);

__global__ void cuda_window(int nt, int nrec, float dt, float ratio,
                            float *data);

__global__ void cuda_normal_misfit(int nrec, float *d_cross_normfact,
                                   float *d_obs_normfact, float *d_cal_normfact,
                                   float *misfit, float *d_weights,
                                   float src_weight);

__global__ void cuda_embed_crop(int nz, int nx, float *d_data, int nz_pad,
                                int nx_pad, float *d_data_pad, bool isEmbed);

__global__ void cuda_spectrum_update(int nf, int nrec,
                                     cuFloatComplex *d_data_obs_F,
                                     cuFloatComplex *d_data_cal_F,
                                     cuFloatComplex *d_source_F,
                                     cuFloatComplex *d_coef);

__global__ void cuda_find_absmax(int n, cuFloatComplex *data, float maxval);

void bp_filter1d(int nSteps, float dt, int nrec, float *d_data, float *filter);

float source_update(int nSteps, float dt, int nrec, float *d_data_obs,
                    float *d_data_cal, float *d_source, cuFloatComplex *d_coef);

void source_update_adj(int nSteps, float dt, int nrec, float *d_data,
                       float amp_ratio, cuFloatComplex *d_coef);

float amp_ratio_comp(int n, float *d_data_obs, float *d_data_cal);

#endif