// Dongzhuo Li 05/06/2018
#include <chrono>
#include <string>
#include "Boundary.h"
#include "Cpml.h"
#include "Model.h"
#include "Parameter.h"
#include "Src_Rec.h"
#include "utilities.h"
using std::string;
// #define VERBOSE
// #define DEBUG
/*        double misfit
        double *grad_Lambda : gradients of Lambda (lame parameter)
        double *grad_Mu : gradients of Mu (shear modulus)
        double *grad_Den : gradients of density
        double *grad_stf : gradients of source time function
        double *Lambda : lame parameter (Mega Pascal)
        double *Mu : shear modulus (Mega Pascal)
        double *Den : density
        double *stf : source time function of all shots
        int calc_id :
                                        calc_id = 0  -- compute residual
                                        calc_id = 1  -- compute gradient
                                        calc_id = 2  -- compute observation only
        int gpu_id  :   CUDA_VISIBLE_DEVICES
        int group_size: number of shots in the group
        int *shot_ids :   processing shot shot_ids
        string para_fname :  parameter path
        // string survey_fname :  survey file (src/rec) path
        // string data_dir : data directory
        // string scratch_dir : temporary files
*/
extern "C" void cufd(float *misfit, float *grad_Lambda, float *grad_Mu,
          float *grad_Den, float *grad_stf, const float *Lambda,
          const float *Mu, const float *Den, const float *stf, int calc_id,
          const int gpu_id, const int group_size, const int *shot_ids,
          const string para_fname) {
  // int deviceCount = 0;
  // CHECK(cudaGetDeviceCount (&deviceCount));
  // printf("number of devices = %d\n", deviceCount);
  CHECK(cudaSetDevice(gpu_id));
  auto start0 = std::chrono::high_resolution_clock::now();

  if (calc_id < 0 || calc_id > 2) {
    printf("Invalid calc_id %d\n", calc_id);
    exit(0);
  }

  // NOTE Read parameter file
  Parameter para(para_fname, calc_id);
  int nz = para.nz();
  int nx = para.nx();
  int nPml = para.nPoints_pml();
  int nPad = para.nPad();
  float dz = para.dz();
  float dx = para.dx();
  float dt = para.dt();
  float f0 = para.f0();

  int iSnap = 500;  // 400
  int nrec = 1;
  float win_ratio = 0.005;
  int nSteps = para.nSteps();
  float amp_ratio = 1.0;

  // transpose models and convert to float
  float *fLambda, *fMu, *fDen;
  fLambda = (float *)malloc(nz * nx * sizeof(float));
  fMu = (float *)malloc(nz * nx * sizeof(float));
  fDen = (float *)malloc(nz * nx * sizeof(float));
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      fLambda[j * nz + i] = Lambda[i * nx + j];
      fMu[j * nz + i] = Mu[i * nx + j];
      fDen[j * nz + i] = Den[i * nx + j];
    }
  }
  Model model(para, fLambda, fMu, fDen);
  // Model model;
  Cpml cpml(para, model);
  Bnd boundaries(para);

  auto startSrc = std::chrono::high_resolution_clock::now();
  Src_Rec src_rec(para, para.survey_fname(), stf, group_size, shot_ids);
  // TODO: group_size -> shot group size
  auto finishSrc = std::chrono::high_resolution_clock::now();
#ifdef VERBOSE
  std::chrono::duration<double> elapsedSrc = finishSrc - startSrc;
  std::cout << "Src_Rec time: " << elapsedSrc.count() << " second(s)"
            << std::endl;
  std::cout << "number of shots " << src_rec.d_vec_z_rec.size() << std::endl;
#endif

  // compute Courant number
  // compCourantNumber(model.h_Cp, nz * nx, dt, dz, dx);

  dim3 threads(TX, TY);
  dim3 blocks((nz + TX - 1) / TX, (nx + TY - 1) / TY);
  // dim3 threads2(TX + 4, TY + 4);
  // dim3 blocks2((nz + TX + 3) / (TX + 4), (nx + TY + 3) / (TY + 4));

  float *d_vz, *d_vx, *d_szz, *d_sxx, *d_sxz, *d_vz_adj, *d_vx_adj, *d_szz_adj,
      *d_sxx_adj, *d_sxz_adj;
  float *d_mem_dvz_dz, *d_mem_dvz_dx, *d_mem_dvx_dz, *d_mem_dvx_dx;
  float *d_mem_dszz_dz, *d_mem_dsxx_dx, *d_mem_dsxz_dz, *d_mem_dsxz_dx;
  float *d_l2Obj_temp_z;
  float *d_l2Obj_temp_x;
  float *h_l2Obj_temp_z = nullptr;
  h_l2Obj_temp_z = (float *)malloc(sizeof(float));
  float *h_l2Obj_temp_x = nullptr;
  h_l2Obj_temp_x = (float *)malloc(sizeof(float));
  float h_l2Obj_z = 0.0;
  float h_l2Obj_x = 0.0;
  float *d_gauss_amp;
  float *d_data_z;
  float *d_data_x;
  float *d_data_obs_z;
  float *d_data_obs_x;
  float *d_res_z;
  float *d_res_x;
  CHECK(cudaMalloc((void **)&d_vz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vz_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vx_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxx_adj, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxz_adj, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_mem_dvz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvz_dx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dx, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_mem_dszz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxx_dx, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxz_dz, nz * nx * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxz_dx, nz * nx * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_l2Obj_temp_z, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_l2Obj_temp_x, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_gauss_amp, 81 * sizeof(float)));
  src_rec_gauss_amp<<<1, threads>>>(d_gauss_amp, 9, 9);

  float *h_snap, *h_snap_back, *h_snap_adj;
  float *h_res_z, *h_res_x;
  h_snap = (float *)malloc(nz * nx * sizeof(float));
  h_snap_back = (float *)malloc(nz * nx * sizeof(float));
  h_snap_adj = (float *)malloc(nz * nx * sizeof(float));

  cudaStream_t *streams = (cudaStream_t *)malloc(group_size * sizeof(cudaStream_t));

  auto finish0 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed0 = finish0 - start0;
#ifdef VERBOSE
  std::cout << "Initialization time: " << elapsed0.count() << " second(s)"
            << std::endl;
#endif

  auto start = std::chrono::high_resolution_clock::now();

  // NOTE Processing Shot
  for (int iShot = 0; iShot < group_size; iShot++) {
#ifdef VERBOSE
    printf("	Processing shot %d\n", shot_ids[iShot]);
#endif
    CHECK(cudaStreamCreate(&streams[iShot]));

    intialArrayGPU<<<blocks, threads>>>(d_vz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_vx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_vz_adj, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_vx_adj, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_szz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_szz_adj, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxx_adj, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_sxz_adj, nz, nx, 0.0);

    intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dx, nz, nx, 0.0);

    intialArrayGPU<<<blocks, threads>>>(d_mem_dszz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxx_dx, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dz, nz, nx, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dx, nz, nx, 0.0);

    nrec = src_rec.vec_nrec.at(iShot);

    CHECK(cudaMalloc((void **)&d_data_z, nrec * nSteps * sizeof(float)));
    intialArrayGPU<<<blocks, threads>>>(d_data_z, nSteps, nrec, 0.0);

    CHECK(cudaMalloc((void **)&d_data_x, nrec * nSteps * sizeof(float)));
    intialArrayGPU<<<blocks, threads>>>(d_data_x, nSteps, nrec, 0.0);

    if (para.if_res()) {
      fileBinLoad(src_rec.vec_data_obs_z.at(iShot), nSteps * nrec,
                  para.data_dir_name() + "/Shot_z" +
                      std::to_string(shot_ids[iShot]) + ".bin");
      fileBinLoad(src_rec.vec_data_obs_x.at(iShot), nSteps * nrec,
                  para.data_dir_name() + "/Shot_x" +
                      std::to_string(shot_ids[iShot]) + ".bin");
      CHECK(cudaMalloc((void **)&d_data_obs_z, nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_data_obs_x, nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res_z, nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res_x, nrec * nSteps * sizeof(float)));
      h_res_z = (float *)malloc(nrec * nSteps * sizeof(float));
      h_res_x = (float *)malloc(nrec * nSteps * sizeof(float));
      intialArrayGPU<<<blocks, threads>>>(d_data_obs_z, nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_data_obs_x, nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_res_z, nSteps, nrec, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_res_x, nSteps, nrec, 0.0);
      CHECK(cudaMemcpyAsync(d_data_obs_z, src_rec.vec_data_obs_z.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));
      CHECK(cudaMemcpyAsync(d_data_obs_x, src_rec.vec_data_obs_x.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));
    }
    // ------------------------ time loop ----------------------------
    for (int it = 0; it <= nSteps - 2; it++) {
      // ================= elastic =====================
      if (para.withAdj()) {
        // save and record from the beginning
        boundaries.field_from_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it);
      }

      // get snapshot at time it
      if (it == iSnap && iShot == 0) {
        CHECK(cudaMemcpy(h_snap, d_vx, nz * nx * sizeof(float),
                         cudaMemcpyDeviceToHost));
      }

      el_stress<<<blocks, threads>>>(
          d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dvz_dz, d_mem_dvz_dx,
          d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
          model.d_ave_Mu, model.d_Den, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x,
          cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
          cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, true, d_szz_adj,
          d_sxx_adj, d_sxz_adj, model.d_LambdaGrad, model.d_MuGrad);

      add_source<<<1, threads>>>(d_szz, d_sxx, src_rec.vec_source.at(iShot)[it],
                                 nz, true, src_rec.vec_z_src.at(iShot),
                                 src_rec.vec_x_src.at(iShot), dt, d_gauss_amp);

      el_velocity<<<blocks, threads>>>(
          d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx,
          d_mem_dsxz_dz, d_mem_dsxx_dx, model.d_Lambda, model.d_Mu,
          model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z, cpml.d_a_z,
          cpml.d_b_z, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
          cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, true, d_vz_adj,
          d_vx_adj, model.d_DenGrad);

      // if (iShot == 0) {
      //     CHECK(cudaMemcpy(h_snap, d_vz, nz * nx * sizeof(float),
      //                      cudaMemcpyDeviceToHost));
      //     fileBinWrite(h_snap, nz * nx, para.data_dir_name() + 
      //                  "/SnapGPU_zy" + std::to_string(it) + ".bin");
      // }
      recording<<<(nrec + 31) / 32, 32>>>(
          d_vz, d_vx, nz, d_data_z, d_data_x, iShot, it + 1, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot));
      
    }  // end of forward time loop

    if (!para.if_res()) {
      CHECK(cudaMemcpyAsync(src_rec.vec_data_z.at(iShot), d_data_z,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_x.at(iShot), d_data_x,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test
    }

#ifdef DEBUG
    fileBinWrite(h_snap, nz * nx, "SnapGPU.bin");
#endif

    // compute residuals
    if (para.if_res()) {
      dim3 blocksT((nSteps + TX - 1) / TX, (nrec + TY - 1) / TY);

      // windowing
      if (para.if_win()) {
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            win_ratio, d_data_obs_z);
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            win_ratio, d_data_z);
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            win_ratio, d_data_obs_x);
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            win_ratio, d_data_x);
      } else {
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio, d_data_obs_z);
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio, d_data_z);
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio, d_data_obs_x);
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio, d_data_x);
      }

      // filtering
      if (para.if_filter()) {
        bp_filter1d(nSteps, dt, nrec, d_data_obs_z, para.filter());
        bp_filter1d(nSteps, dt, nrec, d_data_z, para.filter());
        bp_filter1d(nSteps, dt, nrec, d_data_obs_x, para.filter());
        bp_filter1d(nSteps, dt, nrec, d_data_x, para.filter());
      }

      // Calculate source update and filter calculated data
      if (para.if_src_update()) {
        amp_ratio = source_update(nSteps, dt, nrec, d_data_obs_z, d_data_z,
                          src_rec.d_vec_source.at(iShot), src_rec.d_coef);
        printf("	Source update => Processing shot %d, amp_ratio = %f\n",
               iShot, amp_ratio);
      }
      amp_ratio = 1.0;  // amplitude not used, so set to 1.0

      // objective function
      gpuMinus<<<blocksT, threads>>>(d_res_z, d_data_obs_z, d_data_z, nSteps, nrec);
      gpuMinus<<<blocksT, threads>>>(d_res_x, d_data_obs_x, d_data_x, nSteps, nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp_z, d_res_z, nSteps * nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp_x, d_res_x, nSteps * nrec);
      CHECK(cudaMemcpy(h_l2Obj_temp_z, d_l2Obj_temp_z, sizeof(float),
                       cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_l2Obj_temp_x, d_l2Obj_temp_x, sizeof(float),
                       cudaMemcpyDeviceToHost));
      h_l2Obj_z += h_l2Obj_temp_z[0];
      h_l2Obj_x += h_l2Obj_temp_x[0];

      //  update source again (adjoint)
      // if (para.if_src_update()) {
      //   source_update_adj(nSteps, dt, nrec, d_res, amp_ratio, src_rec.d_coef);
      // }

      // filtering again (adjoint)
      // if (para.if_filter()) {
      //   bp_filter1d(nSteps, dt, nrec, d_res, para.filter());
      // }
      // windowing again (adjoint)
      if (para.if_win()) {
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            win_ratio, d_res_z);
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            win_ratio, d_res_x);
      } else {
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio, d_res_z);
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio, d_res_x);
      }
      
      // if (iShot == 0) { 
      // CHECK(cudaMemcpy(h_res_z, d_res_z, nSteps * src_rec.vec_nrec.at(iShot) * sizeof(float),
      //                    cudaMemcpyDeviceToHost));
      // fileBinWrite(h_res_z, nSteps * src_rec.vec_nrec.at(iShot),
      //              para.data_dir_name() + "/res_z.bin");
      
      // CHECK(cudaMemcpy(h_res_x, d_res_x, nSteps * src_rec.vec_nrec.at(iShot) * sizeof(float),
      //                    cudaMemcpyDeviceToHost));
      // fileBinWrite(h_res_x, nSteps * src_rec.vec_nrec.at(iShot),
      //              para.data_dir_name() + "/res_x.bin");
      // exit(0);
      // }
      CHECK(cudaMemcpyAsync(src_rec.vec_res_z.at(iShot), d_res_z,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_z.at(iShot), d_data_z,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs_z.at(iShot), d_data_obs_z,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // save preconditioned observed

      CHECK(cudaMemcpyAsync(src_rec.vec_res_x.at(iShot), d_res_x,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_x.at(iShot), d_data_x,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs_x.at(iShot), d_data_obs_x,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // save preconditioned observed
      CHECK(cudaMemcpy(src_rec.vec_source.at(iShot),
                       src_rec.d_vec_source.at(iShot), nSteps * sizeof(float),
                       cudaMemcpyDeviceToHost));
    }
    // =================
    cudaDeviceSynchronize();

    if (para.withAdj()) {
      // --------------------- Backward ----------------------------
      // initialization
      intialArrayGPU<<<blocks, threads>>>(d_vz_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_vx_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_szz_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_sxx_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_sxz_adj, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dszz_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dz, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_mem_dsxx_dx, nz, nx, 0.0);
      intialArrayGPU<<<blocks, threads>>>(model.d_StfGrad, nSteps, 1, 0.0);
      initialArray(model.h_StfGrad, nSteps, 0.0);

      el_stress_adj<<<blocks, threads>>>(
            d_vz_adj, d_vx_adj, d_szz_adj, d_sxx_adj, d_sxz_adj, d_mem_dszz_dz,
            d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dvz_dz,
            d_mem_dvz_dx, d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
            model.d_ave_Mu, model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b,
            cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
            cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
            cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad);      
 
      res_injection<<<(nrec + 31) / 32, 32>>>(
          d_vz_adj, d_vx_adj, nz, d_res_z, d_res_x, nSteps - 1, dt, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot));

      el_velocity_adj<<<blocks, threads>>>(
          d_vz_adj, d_vx_adj, d_szz_adj, d_sxx_adj, d_sxz_adj, d_mem_dszz_dz,
          d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dvz_dz,
          d_mem_dvz_dx, d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
          model.d_ave_Mu, model.d_Den, model.d_ave_Byc_a, model.d_ave_Byc_b,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
          cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad);   

      for (int it = nSteps - 2; it >= 0; it--) {
        // source time function kernels
        source_grad<<<1, 1>>>(d_szz_adj, d_sxx_adj, nz, model.d_StfGrad, it, dt,
                              src_rec.vec_z_src.at(iShot),
                              src_rec.vec_x_src.at(iShot));

        el_velocity<<<blocks, threads>>>(
            d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dszz_dz, d_mem_dsxz_dx,
            d_mem_dsxz_dz, d_mem_dsxx_dx, model.d_Lambda, model.d_Mu,
            model.d_ave_Byc_a, model.d_ave_Byc_b, cpml.d_K_z, cpml.d_a_z,
            cpml.d_b_z, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
            cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half,
            cpml.d_a_x_half, cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad,
            false, d_vz_adj, d_vx_adj, model.d_DenGrad);
        boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it, false);

        add_source<<<1, threads>>>(
            d_szz, d_sxx, src_rec.vec_source.at(iShot)[it], nz, false,
            src_rec.vec_z_src.at(iShot), src_rec.vec_x_src.at(iShot), dt,
            d_gauss_amp);

        el_stress<<<blocks, threads>>>(
            d_vz, d_vx, d_szz, d_sxx, d_sxz, d_mem_dvz_dz, d_mem_dvz_dx,
            d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda, model.d_Mu,
            model.d_ave_Mu, model.d_Den, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
            cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x,
            cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
            cpml.d_b_x_half, nz, nx, dt, dz, dx, nPml, nPad, false, d_szz_adj,
            d_sxx_adj, d_sxz_adj, model.d_LambdaGrad, model.d_MuGrad);
        boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_vz, d_vx, it, true);

        el_stress_adj<<<blocks, threads>>>(
            d_vz_adj, d_vx_adj, d_szz_adj, d_sxx_adj, d_sxz_adj, d_mem_dszz_dz,
            d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dvz_dz,
            d_mem_dvz_dx, d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda,
            model.d_Mu, model.d_ave_Mu, model.d_Den, model.d_ave_Byc_a,
            model.d_ave_Byc_b, cpml.d_K_z_half, cpml.d_a_z_half,
            cpml.d_b_z_half, cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half,
            cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, cpml.d_K_x, cpml.d_a_x,
            cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad);

        res_injection<<<(nrec + 31) / 32, 32>>>(
            d_vz_adj, d_vx_adj, nz, d_res_z, d_res_x, it, dt, nSteps, nrec,
            src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot));
            
        el_velocity_adj<<<blocks, threads>>>(
            d_vz_adj, d_vx_adj, d_szz_adj, d_sxx_adj, d_sxz_adj, d_mem_dszz_dz,
            d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dvz_dz,
            d_mem_dvz_dx, d_mem_dvx_dz, d_mem_dvx_dx, model.d_Lambda,
            model.d_Mu, model.d_ave_Mu, model.d_Den, model.d_ave_Byc_a,
            model.d_ave_Byc_b, cpml.d_K_z_half, cpml.d_a_z_half,
            cpml.d_b_z_half, cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half,
            cpml.d_K_z, cpml.d_a_z, cpml.d_b_z, cpml.d_K_x, cpml.d_a_x,
            cpml.d_b_x, nz, nx, dt, dz, dx, nPml, nPad);


        // if (it == iSnap && iShot == 0) {
        //   CHECK(cudaMemcpy(h_snap_back, d_vz, nz * nx * sizeof(float),
        //                    cudaMemcpyDeviceToHost));
        //   CHECK(cudaMemcpy(h_snap_adj, d_szz_adj, nz * nx * sizeof(float),
        //                    cudaMemcpyDeviceToHost));
        // }
        // if (iShot == 51) {
        //   CHECK(cudaMemcpy(h_snap_adj, d_vz_adj, nz * nx * sizeof(float),
        //                    cudaMemcpyDeviceToHost));
        //   fileBinWrite(h_snap_adj, nz * nx, para.data_dir_name() + 
        //                "/SnapGPU_adj_" + std::to_string(it) + ".bin");
        //   CHECK(cudaMemcpy(h_snap, d_vz, nz * nx * sizeof(float),
        //                    cudaMemcpyDeviceToHost));
        //   fileBinWrite(h_snap, nz * nx, para.data_dir_name() + 
        //                "/SnapGPU_" + std::to_string(it) + ".bin");
        // }
      }  // the end of backward time loop

#ifdef DEBUG
      fileBinWrite(h_snap_back, nz * nx, "SnapGPU_back.bin");
      fileBinWrite(h_snap_adj, nz * nx, "SnapGPU_adj.bin");
#endif

      // transfer source gradient to cpu
      CHECK(cudaMemcpy(model.h_StfGrad, model.d_StfGrad, nSteps * sizeof(float),
                       cudaMemcpyDeviceToHost));
      for (int it = 0; it < nSteps; it++) {
        grad_stf[iShot * nSteps + it] = model.h_StfGrad[it];
      }
    }  // end bracket of if adj
    CHECK(cudaFree(d_data_z));
    CHECK(cudaFree(d_data_x));
    if (para.if_res()) {
      CHECK(cudaFree(d_data_obs_z));
      CHECK(cudaFree(d_data_obs_x));
      CHECK(cudaFree(d_res_z));
      CHECK(cudaFree(d_res_x));
    }
  }  // the end of shot loop

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
#ifdef VERBOSE
  std::cout << "Elapsed time: " << elapsed.count() << " second(s)."
            << std::endl;
#endif

  if (para.withAdj()) {
    // transfer gradients to cpu
    CHECK(cudaMemcpy(model.h_LambdaGrad, model.d_LambdaGrad,
                     nz * nx * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(model.h_MuGrad, model.d_MuGrad, nz * nx * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(model.h_DenGrad, model.d_DenGrad, nz * nx * sizeof(float),
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < nz; i++) {
      for (int j = 0; j < nx; j++) {
        grad_Lambda[i * nx + j] = model.h_LambdaGrad[j * nz + i];
        grad_Mu[i * nx + j] = model.h_MuGrad[j * nz + i];
        grad_Den[i * nx + j] = model.h_DenGrad[j * nz + i];
      }
    }
#ifdef DEBUG
    fileBinWrite(model.h_LambdaGrad, nz * nx, "LambdaGradient.bin");
    fileBinWrite(model.h_MuGrad, nz * nx, "MuGradient.bin");
    fileBinWrite(model.h_DenGrad, nz * nx, "DenGradient.bin");
#endif

    // if (para.if_save_scratch()) {
    //   for (int iShot = 0; iShot < group_size; iShot++) {
    //     fileBinWrite(src_rec.vec_res.at(iShot),
    //                  nSteps * src_rec.vec_nrec.at(iShot),
    //                  para.scratch_dir_name() + "/Residual_Shot" +
    //                      std::to_string(shot_ids[iShot]) + ".bin");
    //     fileBinWrite(src_rec.vec_data.at(iShot),
    //                  nSteps * src_rec.vec_nrec.at(iShot),
    //                  para.scratch_dir_name() + "/Syn_Shot" +
    //                      std::to_string(shot_ids[iShot]) + ".bin");
    //     fileBinWrite(src_rec.vec_data_obs.at(iShot),
    //                  nSteps * src_rec.vec_nrec.at(iShot),
    //                  para.scratch_dir_name() + "/CondObs_Shot" +
    //                      std::to_string(shot_ids[iShot]) + ".bin");
    //       fileBinWrite(src_rec.vec_source.at(iShot), nSteps,
    //                    para.scratch_dir_name() + "/src_updated" +
    //                        std::to_string(shot_ids[iShot]) + ".bin");
    //   }
    // }
  }

  if (!para.if_res()) {
    startSrc = std::chrono::high_resolution_clock::now();
    for (int iShot = 0; iShot < group_size; iShot++) {
      fileBinWrite(src_rec.vec_data_z.at(iShot),
                   nSteps * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_z" +
                       std::to_string(shot_ids[iShot]) + ".bin");
      fileBinWrite(src_rec.vec_data_x.at(iShot),
                   nSteps * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_x" +
                       std::to_string(shot_ids[iShot]) + ".bin");
    }
    finishSrc = std::chrono::high_resolution_clock::now();
#ifdef VERBOSE
    elapsedSrc = finishSrc - startSrc;
    std::cout << "Obs data saving time: " << elapsedSrc.count() << " second(s)"
              << std::endl;
#endif
  }
  if (para.if_res()) {
    h_l2Obj_z = 0.5 * h_l2Obj_z;
    h_l2Obj_x = 0.5 * h_l2Obj_x;
#ifdef VERBOSE
    std::cout << "Total l2 residual z = " << std::to_string(h_l2Obj_z) << std::endl;
    std::cout << "Total l2 residual x = " << std::to_string(h_l2Obj_x) << std::endl;
    std::cout << "calc_id = " << calc_id << std::endl;
#endif
    *misfit = h_l2Obj_z + h_l2Obj_x;
  }

  free(h_l2Obj_temp_z);
  free(h_l2Obj_temp_x);

  free(h_snap);

  free(h_snap_back);

  free(h_snap_adj);

  free(fLambda);

  free(fMu);

  free(fDen);

  // destroy the streams
  for (int iShot = 0; iShot < group_size; iShot++)
    CHECK(cudaStreamDestroy(streams[iShot]));

  cudaFree(d_vz);
  cudaFree(d_vx);
  cudaFree(d_szz);
  cudaFree(d_sxx);
  cudaFree(d_sxz);
  cudaFree(d_vz_adj);
  cudaFree(d_vx_adj);
  cudaFree(d_szz_adj);
  cudaFree(d_sxx_adj);
  cudaFree(d_sxz_adj);
  cudaFree(d_mem_dvz_dz);
  cudaFree(d_mem_dvz_dx);
  cudaFree(d_mem_dvx_dz);
  cudaFree(d_mem_dvx_dx);
  cudaFree(d_mem_dszz_dz);
  cudaFree(d_mem_dsxx_dx);
  cudaFree(d_mem_dsxz_dz);
  cudaFree(d_mem_dsxz_dx);
  cudaFree(d_l2Obj_temp_z);
  cudaFree(d_l2Obj_temp_x);
  cudaFree(d_gauss_amp);

#ifdef VERBOSE
  std::cout << "Done!" << std::endl;
#endif
}
