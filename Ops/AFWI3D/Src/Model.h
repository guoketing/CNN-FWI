// Dongzhuo Li 04/20/2018
#ifndef MODEL_H__
#define MODEL_H__

#include "Parameter.h"

// void fileBinLoad(float *h_bin, int size, std::string fname);
// void intialArray(float *ip, int size, float value);
// __global__ void moduliInit(float *d_Cp, float *d_Cs, float *d_Den, float
// *d_Lambda, float *d_Mu, int nx, int ny);

class Model {
 private:
  int nz_, nx_, ny_, nShape;

 public:
  Model();
  Model(const Parameter &para);
  Model(const Parameter &para, const float *Vp_);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model();

  float *h_Vp;
  float *d_Vp;

  float *h_VpGrad;
  float *d_VpGrad;
  float *h_StfGrad;
  float *d_StfGrad;

  int nz() { return nz_; }
  int nx() { return nx_; }
  int ny() { return ny_; }
};

#endif