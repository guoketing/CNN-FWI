#include <iostream>
#include <string>
#include "Model.h"
#include "Parameter.h"
#include "utilities.h"

// model default constructor
Model::Model() {
  std::cout << "ERROR: You need to supply parameters to initialize models!"
            << std::endl;
  exit(1);
}

// model constructor from parameter file
Model::Model(const Parameter &para, const float *Vp_) {
  nz_ = para.nz();
  nx_ = para.nx();
  ny_ = para.ny();
  nShape = nz_ * nx_ * ny_;

  dim3 threads(TX, TY, TZ);
  dim3 blocks((nz_ + TX - 1) / TX, (nx_ + TY - 1) / TY, (ny_ + TZ - 1) / TZ);

  dim3 threads_2d(32, 16);
  dim3 blocks_2d((para.nSteps() + 32 - 1) / 32, 1);

  h_Vp = (float *)malloc(nShape * sizeof(float));
  h_VpGrad = (float *)malloc(nShape * sizeof(float));
  h_StfGrad = (float *)malloc(para.nSteps() * sizeof(float));

  // load Vp, Vs, and Den binaries
  
  for (int i = 0; i < nShape; i++) {
    if (Vp_[i] < 0.0) {
      printf("Vp is negative!!!");
      exit(1);
    }
    h_Vp[i] = Vp_[i];
  }

  initialArray(h_VpGrad, nShape, 0.0);
  initialArray(h_StfGrad, para.nSteps(), 0.0);

  CHECK(cudaMalloc((void **)&d_Vp, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_VpGrad, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_StfGrad, para.nSteps() * sizeof(float)));

  intialArrayGPU<<<blocks, threads>>>(d_VpGrad, nz_, nx_, ny_, 0.0);
  intial2dArrayGPU<<<blocks_2d, threads_2d>>>(d_StfGrad, para.nSteps(), 1, 0.0);

  CHECK(cudaMemcpy(d_Vp, h_Vp, nShape * sizeof(float), cudaMemcpyHostToDevice));

}

Model::~Model() {
  free(h_Vp);
  free(h_VpGrad);
  free(h_StfGrad);
  CHECK(cudaFree(d_Vp));
  CHECK(cudaFree(d_VpGrad));
  CHECK(cudaFree(d_StfGrad));
}