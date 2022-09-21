#ifndef BND_H__
#define BND_H__

#include "Parameter.h"

class Bnd {
 private:
  int nz_, nx_, ny_, nPml_, nPad_, nSteps_, nzBnd_, nxBnd_, nyBnd_, len_Bnd_vec_,
      nLayerStore_;

  bool isAc_, withAdj_;

 public:
  float *d_Bnd_u;

  Bnd(const Parameter &para);
  Bnd(const Bnd&) = delete;
  Bnd& operator=(const Bnd&) = delete;

  ~Bnd();

  void field_from_bnd(float *d_u, int indT);

  void field_to_bnd(float *d_u, int indT);

  int len_Bnd_vec() { return len_Bnd_vec_; }
};

#endif