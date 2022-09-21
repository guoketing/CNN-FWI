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

#define VERBOSE
// #define DEBUG
/*
        double misfit
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
extern "C" void cufd(float *misfit, float *grad_Vp, float *grad_stf, const float *Vp,
          const float *stf, int calc_id,
          const int gpu_id, const int group_size, const int *shot_ids,
          const string para_fname) {
  // int deviceCount = 0;
  // CHECK(cudaGetDeviceCount (&deviceCount));
  // printf("number of devices = %d\n", deviceCount);
  CHECK(cudaSetDevice(gpu_id));
  auto start0 = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE
  printf("Calc_ID = %d\n", calc_id);
#endif
  if (calc_id < 0 || calc_id > 2) {
    printf("Invalid calc_id %d\n", calc_id);
    exit(0);
  }

  // NOTE Read parameter file
  Parameter para(para_fname, calc_id);
  int nz = para.nz();
  int nx = para.nx();
  int ny = para.ny();
  int nShape = nz * nx * ny;
  int nPml = para.nPoints_pml();
  int nPad = para.nPad();
  float dz = para.dz();
  float dx = para.dx();
  float dy = para.dy();
  float dt = para.dt();
  float f0 = para.f0();


  int iSnap = 500;  // 400
  int nrec = 1;
  float win_ratio = 0.005;
  int nSteps = para.nSteps();
  float amp_ratio = 1.0;

  // transpose models and convert to float
  float *fVp;
  fVp = (float *)malloc(nShape * sizeof(float));
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      for (int k = 0; k < ny; k++) {
        fVp[k * (nx*nz) + j*nz + i] = Vp[i * (nx*ny) + j * (ny) + k];
      }
    }
  }
  Model model(para, fVp);
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

  dim3 threads(TX, TY, TZ);
  dim3 blocks((nz + TX - 1) / TX, (nx + TY - 1) / TY, (ny + TZ - 1) / TZ);

  float *d_un, *d_up, *d_uc, *d_un_adj, *d_up_adj, *d_uc_adj;
  float *d_phiz, *d_phix, *d_phiy, *d_psci;
  float *d_l2Obj_temp;
  float *h_l2Obj_temp = nullptr;
  h_l2Obj_temp = (float *)malloc(sizeof(float));
  float h_l2Obj = 0.0;
  float *d_gauss_amp;
  float *d_data;
  float *d_data_obs;
  float *d_res;

  float *d_obs_normfact, *d_cal_normfact, *d_cross_normfact;
  CHECK(cudaMalloc((void **)&d_un, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_up, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_uc, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_phiz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_phix, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_phiy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_psci, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_un_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_up_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_uc_adj, nShape * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_l2Obj_temp, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_gauss_amp, 729 * sizeof(float)));
  dim3 blocks3(1, 1, (9 + TZ - 1) / TZ);
  src_rec_gauss_amp<<<blocks3, threads>>>(d_gauss_amp, 9, 9, 9);

  float *h_snap, *h_snap_back, *h_snap_adj;
  h_snap = (float *)malloc(nShape * sizeof(float));
  h_snap_back = (float *)malloc(nShape * sizeof(float));
  h_snap_adj = (float *)malloc(nShape * sizeof(float));

  cudaStream_t *streams = (cudaStream_t *)malloc(group_size * sizeof(cudaStream_t));
    printf("a");
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
  
    intialArrayGPU<<<blocks, threads>>>(d_up, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_uc, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_un, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_phiz, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_phix, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_phiy, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_psci, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_up_adj, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_uc_adj, nz, nx, ny, 0.0);
    intialArrayGPU<<<blocks, threads>>>(d_un_adj, nz, nx, ny, 0.0);

    nrec = src_rec.vec_nrec.at(iShot);

    dim3 threads2(32, 32);
    dim3 blocks2((nSteps + 32 - 1) / 32, (nrec + 32 - 1) / 32);

    CHECK(cudaMalloc((void **)&d_data, nrec * nSteps * sizeof(float)));
    intial2dArrayGPU<<<blocks2, threads2>>>(d_data, nSteps, nrec, 0.0);
    if (para.if_res()) {
      fileBinLoad(src_rec.vec_data_obs.at(iShot), nSteps * nrec,
                  para.data_dir_name() + "/Shot" +
                      std::to_string(shot_ids[iShot]) + ".bin");
      CHECK(cudaMalloc((void **)&d_data_obs, nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res, nrec * nSteps * sizeof(float)));
      intial2dArrayGPU<<<blocks2, threads2>>>(d_data_obs, nSteps, nrec, 0.0);
      intial2dArrayGPU<<<blocks2, threads2>>>(d_res, nSteps, nrec, 0.0);
      CHECK(cudaMemcpyAsync(d_data_obs, src_rec.vec_data_obs.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));
            // initialize normalization factors
      if (para.if_cross_misfit()) {
        CHECK(cudaMalloc((void **)&d_obs_normfact, nrec * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_cal_normfact, nrec * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_cross_normfact, nrec * sizeof(float)));
        intial2dArrayGPU<<<1, 512>>>(d_obs_normfact, nrec, 1, 0.0);
        intial2dArrayGPU<<<1, 512>>>(d_cal_normfact, nrec, 1, 0.0);
        intial2dArrayGPU<<<1, 512>>>(d_cross_normfact, nrec, 1, 0.0);
      }
    }
    // ------------------------ time loop ----------------------------
    for (int it = 0; it < nSteps - 1; it++) {
      // =========================== acoustic ==================================

      // get snapshot at time it
        if (it == 700 && iShot == 0) {
         CHECK(cudaMemcpy(h_snap, d_up, nShape * sizeof(float), cudaMemcpyDeviceToHost));
         fileBinWrite(h_snap, nShape, "SnapGPU_for_.bin");
        printf("saved the forward wavefield!");
      }

      if (para.withAdj()) {
         // save and record from the beginning
         boundaries.field_from_bnd(d_up, it);
      }

      propagate<<<blocks, threads>>>(
                d_un, d_uc, d_up, d_phiz, d_phix, d_phiy, d_psci,
                model.d_Vp, cpml.d_sigma_z, cpml.d_sigma_x,
                cpml.d_sigma_y, nz, nx, ny, dt, dz, dx, dy, nPml, true);

      add_source<<<blocks3, threads>>>(d_un, src_rec.vec_source.at(iShot)[it],
                                 nz, nx, true, src_rec.vec_z_src.at(iShot),
                                 src_rec.vec_x_src.at(iShot), src_rec.vec_y_src.at(iShot), dt, d_gauss_amp);

      exchange_wavefield<<<blocks, threads>>>(d_un, d_uc, d_up, nz, nx);

      recording<<<(nrec + 31) / 32, 32>>>(
            d_uc, nz, nx, d_data, iShot, it + 1, nSteps, nrec,
            src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot), src_rec.d_vec_y_rec.at(iShot));
    }  // end of forward time loop

    if (!para.if_res()) {
      CHECK(cudaMemcpyAsync(src_rec.vec_data.at(iShot), d_data,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test
    }

      #ifdef DEBUG
    fileBinWrite(h_snap, nz * nx * ny, "SnapGPU.bin");
      #endif

    // compute residuals
    if (para.if_res()) {
      dim3 blocksT((nSteps + TX - 1) / TX, (nrec + TY - 1) / TY);

      // windowing
      if (para.if_win()) {
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_data_obs);
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_data);
      } else {
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio,
                                          d_data_obs);
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio, d_data);
      }

      // filtering
      if (para.if_filter()) {
        bp_filter1d(nSteps, dt, nrec, d_data_obs, para.filter());
        bp_filter1d(nSteps, dt, nrec, d_data, para.filter());
      }

      // normalization 09/26/2019
      if (para.if_cross_misfit()) {
        cuda_find_normfact<<<nrec, 512>>>(nSteps, nrec, d_data_obs, d_data_obs,
                                          d_obs_normfact);
        cuda_find_normfact<<<nrec, 512>>>(nSteps, nrec, d_data, d_data,
                                          d_cal_normfact);
        cuda_find_normfact<<<nrec, 512>>>(nSteps, nrec, d_data_obs, d_data,
                                          d_cross_normfact);
      }

      // Calculate source update and filter calculated data
      if (para.if_src_update()) {
        amp_ratio =
            source_update(nSteps, dt, nrec, d_data_obs, d_data,
                          src_rec.d_vec_source.at(iShot), src_rec.d_coef);
        printf("	Source update => Processing shot %d, amp_ratio = %f\n",
               iShot, amp_ratio);
      }
      amp_ratio = 1.0;  // amplitude not used, so set to 1.0

      // objective function
      if (!para.if_cross_misfit()) {
        gpuMinus<<<blocksT, threads>>>(d_res, d_data_obs, d_data, nSteps, nrec);
        cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp, d_res, nSteps * nrec);
      } else {
        cuda_normal_misfit<<<1, 512>>>(nrec, d_cross_normfact, d_obs_normfact,
                                       d_cal_normfact, d_l2Obj_temp,
                                       src_rec.d_vec_weights.at(iShot),
                                       src_rec.vec_srcweights.at(iShot));
      }

      CHECK(cudaMemcpy(h_l2Obj_temp, d_l2Obj_temp, sizeof(float),
                       cudaMemcpyDeviceToHost));
      h_l2Obj += h_l2Obj_temp[0];

      //  update source again (adjoint)
      if (para.if_src_update()) {
        source_update_adj(nSteps, dt, nrec, d_res, amp_ratio, src_rec.d_coef);
      }

      // compute negative adjoint source for the normalization objective
      // function 09/26/2019
      if (para.if_cross_misfit()) {
        cuda_normal_adjoint_source<<<blocksT, threads>>>(
            nSteps, nrec, d_obs_normfact, d_cal_normfact, d_cross_normfact,
            d_data_obs, d_data, d_res, src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot));
      }

      // filtering again (adjoint)
      if (para.if_filter()) {
        bp_filter1d(nSteps, dt, nrec, d_res, para.filter());
      }
      // windowing again (adjoint)
      if (para.if_win()) {
        cuda_window<<<blocksT, threads>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_res);
      } else {
        cuda_window<<<blocksT, threads>>>(nSteps, nrec, dt, win_ratio, d_res);
      }

      CHECK(cudaMemcpyAsync(src_rec.vec_res.at(iShot), d_res,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test

      CHECK(cudaMemcpyAsync(src_rec.vec_data.at(iShot), d_data,
                            nSteps * nrec * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs.at(iShot), d_data_obs,
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
      intialArrayGPU<<<blocks, threads>>>(d_un_adj, nz, nx, ny, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_up_adj, nz, nx, ny, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_uc_adj, nz, nx, ny, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_phiz, nz, nx, ny, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_phix, nz, nx, ny, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_phix, nz, nx, ny, 0.0);
      intialArrayGPU<<<blocks, threads>>>(d_psci, nz, nx, ny, 0.0);
      intial2dArrayGPU<<<blocks, threads>>>(model.d_StfGrad, nSteps, 1, 0.0);
      initialArray(model.h_StfGrad, nSteps, 0.0);

      propagate<<<blocks, threads>>>(
          d_un_adj, d_uc_adj, d_up_adj, d_phiz, d_phix, d_phiy, d_psci,
          model.d_Vp, cpml.d_sigma_z, cpml.d_sigma_x, cpml.d_sigma_y, nz, nx, ny, dt, dz, dx, dy, nPml, true);

      res_injection<<<(nrec + 31) / 32, 32>>>(
          d_un_adj, nz, nx, d_res, nSteps - 1, dt, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot), src_rec.d_vec_y_rec.at(iShot));

      exchange_wavefield<<<blocks, threads>>>(d_un_adj, d_uc_adj, d_up_adj, nz, nx);

      image_condition<<<blocks, threads>>>(d_uc, d_uc_adj, nz, nx, ny, dz, dx, dy, nPml, model.d_Vp, model.d_VpGrad);

      for (int it = nSteps - 2; it >= 0; it--) {
        // source time function kernels
        // source_grad<<<1, 1>>>(d_szz_adj, d_sxx_adj, nz, model.d_StfGrad, it, dt,
        //                       src_rec.vec_z_src.at(iShot),
        //                       src_rec.vec_x_src.at(iShot),
        //                       src_rec.vec_src_rxz.at(iShot));

        exchange_wavefield<<<blocks, threads>>>(d_up, d_uc, d_un, nz, nx);

        add_source<<<blocks3, threads>>>(d_un, src_rec.vec_source.at(iShot)[it],
                                 nz, nx, false, src_rec.vec_z_src.at(iShot),
                                 src_rec.vec_x_src.at(iShot), src_rec.vec_y_src.at(iShot), dt, d_gauss_amp);

        propagate<<<blocks, threads>>>(
                d_un, d_uc, d_up, d_phiz, d_phix, d_phiy, d_psci,
                model.d_Vp, cpml.d_sigma_z, cpml.d_sigma_x, cpml.d_sigma_y, nz, nx, ny, dt, dz, dx, dy, nPml, false);

        boundaries.field_to_bnd(d_up, it);


        if (it == 700 && iShot == 0) {
                 CHECK(cudaMemcpy(h_snap, d_up, nShape * sizeof(float), cudaMemcpyDeviceToHost));
                 fileBinWrite(h_snap, nShape, "SnapGPU_back_.bin");
                 CHECK(cudaMemcpy(h_snap_adj, d_up_adj, nShape * sizeof(float), cudaMemcpyDeviceToHost));
                 fileBinWrite(h_snap_adj, nShape, "SnapGPU_adj_.bin");
                 printf("saved the backward wavefield!");
                exit(0);
        }


        propagate<<<blocks, threads>>>(
          d_un_adj, d_uc_adj, d_up_adj, d_phiz, d_phix, d_phiy, d_psci,
          model.d_Vp, cpml.d_sigma_z, cpml.d_sigma_x, cpml.d_sigma_y, nz, nx, ny, dt, dz, dx, dy, nPml, true);

        res_injection<<<(nrec + 31) / 32, 32>>>(
          d_un_adj, nz, nx, d_res, it, dt, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot), src_rec.d_vec_y_rec.at(iShot));

        exchange_wavefield<<<blocks, threads>>>(d_un_adj, d_uc_adj, d_up_adj, nz, nx);

        image_condition<<<blocks, threads>>>(d_uc, d_uc_adj, nz, nx, ny, dz, dx, dy, nPml, model.d_Vp, model.d_VpGrad);

        // if (it == iSnap && iShot == 0) {
        //   CHECK(cudaMemcpy(h_snap_back, d_up, nShape * sizeof(float),
        //                    cudaMemcpyDeviceToHost));
        //   CHECK(cudaMemcpy(h_snap_adj, d_up_adj, nShape * sizeof(float),
        //                    cudaMemcpyDeviceToHost));
        // }
        /*
        if (it % 10 == 0 && iShot == 0) {
          CHECK(cudaMemcpy(h_snap_adj, d_up_adj, nShape * sizeof(float),
                            cudaMemcpyDeviceToHost));
           fileBinWrite(h_snap_adj, nShape,
                        "SnapGPU_adj_" + std::to_string(it) + ".bin");
           CHECK(cudaMemcpy(h_snap, d_up, nShape * sizeof(float),
                            cudaMemcpyDeviceToHost));
           fileBinWrite(h_snap, nShape,
                       "SnapGPU_" + std::to_string(it) + ".bin");
        }
        */
      }  // the end of backward time loop
      #ifdef DEBUG
        fileBinWrite(h_snap_back, nShape, "SnapGPU_back.bin");
        fileBinWrite(h_snap_adj, nShape, "SnapGPU_adj.bin");
      #endif
      // transfer source gradient to cpu
      CHECK(cudaMemcpy(model.h_StfGrad, model.d_StfGrad, nSteps * sizeof(float),
                       cudaMemcpyDeviceToHost));
      for (int it = 0; it < nSteps; it++) {
        grad_stf[iShot * nSteps + it] = model.h_StfGrad[it];
      }
    }  // end bracket of if adj
    CHECK(cudaFree(d_data));
    if (para.if_res()) {
      CHECK(cudaFree(d_data_obs));
      CHECK(cudaFree(d_res));
       if (para.if_cross_misfit()) {
         CHECK(cudaFree(d_obs_normfact));
         CHECK(cudaFree(d_cal_normfact));
         CHECK(cudaFree(d_cross_normfact));
       }
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
    CHECK(cudaMemcpy(model.h_VpGrad, model.d_VpGrad,
                     nShape * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < nz; i++) {
      for (int j = 0; j < nx; j++) {
        for (int k = 0; k < ny; k++){
        grad_Vp[i * (nx*ny) + j * (ny) + k] = model.h_VpGrad[k * (nz * nx) + j * nz + i];
          }
      }
    }
#ifdef DEBUG
    fileBinWrite(model.h_VpGrad, nShape, "VpGradient.bin");
#endif
  }

  if (!para.if_res()) {
    startSrc = std::chrono::high_resolution_clock::now();
    for (int iShot = 0; iShot < group_size; iShot++) {

      fileBinWrite(src_rec.vec_data.at(iShot),
                   nSteps * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot" +
                       std::to_string(shot_ids[iShot]) + ".bin");
    }
    finishSrc = std::chrono::high_resolution_clock::now();
#ifdef VERBOSE
    elapsedSrc = finishSrc - startSrc;
    std::cout << "Obs data saving time: " << elapsedSrc.count() << " second(s)"
              << std::endl;
#endif
  }

  // #ifdef DEBUG
  //   std::cout << "cufd--" << __LINE__ << std::endl;
  // #endif

  // output residual
  if (para.if_res()) {
    h_l2Obj = 0.5 * h_l2Obj;  // DL 02/21/2019 (need to make misfit accurate
                              // here rather than in the script)
#ifdef VERBOSE
    std::cout << "Total l2 residual = " << std::to_string(h_l2Obj) << std::endl;
    std::cout << "calc_id = " << calc_id << std::endl;
#endif
    *misfit = h_l2Obj;
  }

  free(h_l2Obj_temp);

  free(h_snap);

  free(h_snap_back);

  free(h_snap_adj);

  free(fVp);

  // destroy the streams
  for (int iShot = 0; iShot < group_size; iShot++)
    CHECK(cudaStreamDestroy(streams[iShot]));

  cudaFree(d_un);
  cudaFree(d_uc);
  cudaFree(d_up);
  cudaFree(d_phiz);
  cudaFree(d_phix);
  cudaFree(d_phiy);
  cudaFree(d_psci);
  cudaFree(d_up_adj);
  cudaFree(d_uc_adj);
  cudaFree(d_un_adj);
  cudaFree(d_l2Obj_temp);
  cudaFree(d_gauss_amp);

#ifdef VERBOSE
  std::cout << "Done!" << std::endl;
#endif
}
