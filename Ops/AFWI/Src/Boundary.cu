#include "Boundary.h"
#include "Parameter.h"
#include "utilities.h"

Bnd::Bnd(const Parameter &para) {
  withAdj_ = para.withAdj();
  if (withAdj_) {
    nz_ = para.nz();
    nx_ = para.nx();
    nPml_ = para.nPoints_pml();
    nPad_ = para.nPad();
    nSteps_ = para.nSteps();

    nzBnd_ = nz_ - 2 * nPml_ - nPad_ + 4;
    nxBnd_ = nx_ - 2 * nPml_ + 4;
    nLayerStore_ = 5;

    len_Bnd_vec_ =
        2 * (nLayerStore_ * nzBnd_ + nLayerStore_ * nxBnd_);  // store n layers

    // allocate the boundary vector in the device
    CHECK(cudaMalloc((void **)&d_Bnd_u,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));
  }
}

Bnd::~Bnd() {
  if (withAdj_) {
    CHECK(cudaFree(d_Bnd_u));
  }
}

void Bnd::field_from_bnd(float *d_u, int indT) {
  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_u, d_Bnd_u, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);
}

void Bnd::field_to_bnd(float *d_u, int indT) {
    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_u, d_Bnd_u, nz_, nx_, nzBnd_,
                                             nxBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nPad_, nSteps_);
}