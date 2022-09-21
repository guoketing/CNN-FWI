#include <stdio.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "libCUFD.h"
#include "omp.h"
// using namespace std;

std::vector<torch::Tensor> fwi_forward(const torch::Tensor &th_Vp,
                                       const torch::Tensor &th_stf, int gpu_id,
                                       const torch::Tensor &th_shot_ids,
                                       const string para_fname) {
  float *misfit_ptr = nullptr;
  misfit_ptr = (float *)malloc(sizeof(float));
  // transform from torch tensor to 1D native array
  auto Vp = th_Vp.data_ptr<float>();
  auto stf = th_stf.data_ptr<float>();
  auto shot_ids = th_shot_ids.data_ptr<int>();
  const int group_size = th_shot_ids.size(0);
  cufd(misfit_ptr, nullptr, nullptr, Vp, stf, 0, gpu_id,
       group_size, shot_ids, para_fname);
  torch::Tensor th_misfit = torch::from_blob(misfit_ptr, {1});
  return {th_misfit.clone()};
}

std::vector<torch::Tensor> fwi_backward(const torch::Tensor &th_Vp,
                                        const torch::Tensor &th_stf, int ngpu,
                                        const torch::Tensor &th_shot_ids,
                                        const string para_fname) {
  const int nz = th_Vp.size(0);
  const int nx = th_Vp.size(1);
  const int ny = th_Vp.size(2);
  const int nSrc = th_stf.size(0);
  const int nSteps = th_stf.size(1);
  const int group_size = th_shot_ids.size(0);
  if (ngpu > group_size) {
    printf("The number of GPUs should be smaller than the number of shots!\n");
    exit(1);
  }
  // transform from torch tensor to 1D native array
  auto Vp = th_Vp.data_ptr<float>();
  auto stf = th_stf.data_ptr<float>();
  auto sepBars = torch::linspace(0, group_size, ngpu + 1,
                                 torch::TensorOptions().dtype(torch::kFloat32));
  std::vector<float> vec_misfit(ngpu);
  std::vector<torch::Tensor> vec_grad_Vp(ngpu);
  std::vector<torch::Tensor> vec_grad_stf(ngpu);
  auto th_grad_Vp_sum = torch::zeros_like(th_Vp);
  float misfit_sum = 0.0;

#pragma omp parallel for num_threads(ngpu)
  for (int i = 0; i < ngpu; i++) {
    // float *misfit_ptr = nullptr;
    // misfit_ptr = (float *)malloc(sizeof(float));
    float misfit = 0.0;
    auto th_grad_Vp = torch::zeros_like(th_Vp);
    auto th_grad_stf = torch::zeros_like(th_stf);
    int startBar = round(sepBars[i].item<int>());
    int endBar = round(sepBars[i + 1].item<int>());
    auto th_sub_shot_ids = th_shot_ids.narrow(0, startBar, endBar - startBar);
    auto shot_ids = th_sub_shot_ids.data_ptr<int>();
    cufd(&misfit, th_grad_Vp.data_ptr<float>(),
         th_grad_stf.data_ptr<float>(), Vp, stf, 1, i,
         th_sub_shot_ids.size(0), shot_ids, para_fname);
    vec_grad_Vp.at(i) = th_grad_Vp;
    vec_grad_stf.at(i) = th_grad_stf;
    // torch::Tensor th_misfit = torch::from_blob(&misfit, {1});
    vec_misfit.at(i) = misfit;
  }
  for (int i = 0; i < ngpu; i++) {
    th_grad_Vp_sum += vec_grad_Vp.at(i);
    misfit_sum += vec_misfit.at(i);
  }
  return {torch::tensor({misfit_sum}), th_grad_Vp_sum, vec_grad_stf.at(0)};
}

void fwi_obscalc(const torch::Tensor &th_Vp, const torch::Tensor &th_stf, int ngpu,
                 const torch::Tensor &th_shot_ids, const string para_fname) {
  const int group_size = th_shot_ids.size(0);
  if (ngpu > group_size) {
    printf("The number of GPUs should be smaller than the number of shots!\n");
    exit(1);
  }
  // transform from torch tensor to 1D native array
  auto Vp = th_Vp.data_ptr<float>();
  auto stf = th_stf.data_ptr<float>();
  auto sepBars = torch::linspace(0, group_size, ngpu + 1,
                                 torch::TensorOptions().dtype(torch::kFloat32));
#pragma omp parallel for num_threads(ngpu)
  for (int i = 0; i < ngpu; i++) {
    int startBar = round(sepBars[i].item<int>());
    int endBar = round(sepBars[i + 1].item<int>());
    auto th_sub_shot_ids = th_shot_ids.narrow(0, startBar, endBar - startBar);
    auto shot_ids = th_sub_shot_ids.data_ptr<int>();
    cufd(nullptr, nullptr, nullptr, Vp, stf, 2, i, th_sub_shot_ids.size(0), shot_ids, para_fname);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwi_forward, "forward");
  m.def("backward", &fwi_backward, "backward");
  m.def("obscalc", &fwi_obscalc, "obscalc");
}