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

  dim3 threads(32, 16);
  dim3 blocks((nz_ + 32 - 1) / 32, (nx_ + 16 - 1) / 16);

  h_Vp = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_VpGrad = (float *)malloc(nz_ * nx_ * sizeof(float));
  h_StfGrad = (float *)malloc(para.nSteps() * sizeof(float));

  // load Vp, Vs, and Den binaries
  
  for (int i = 0; i < nz_ * nx_; i++) {
    if (Vp_[i] < 0.0) {
      // h_Vp[i] = 0;
      printf("Vp is negative!!!");
      exit(1);
    }
    h_Vp[i] = Vp_[i];
  }

  initialArray(h_VpGrad, nz_ * nx_, 0.0);
  initialArray(h_StfGrad, para.nSteps(), 0.0);

  CHECK(cudaMalloc((void **)&d_Vp, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_VpGrad, nz_ * nx_ * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_StfGrad, para.nSteps() * sizeof(float)));

  intialArrayGPU<<<blocks, threads>>>(d_VpGrad, nz_, nx_, 0.0);
  intialArrayGPU<<<blocks, threads>>>(d_StfGrad, para.nSteps(), 1, 0.0);

  CHECK(cudaMemcpy(d_Vp, h_Vp, nz_ * nx_ * sizeof(float),
                   cudaMemcpyHostToDevice));

}

Model::~Model() {
  free(h_Vp);
  free(h_VpGrad);
  free(h_StfGrad);
  CHECK(cudaFree(d_Vp));
  CHECK(cudaFree(d_VpGrad));
  CHECK(cudaFree(d_StfGrad));
}