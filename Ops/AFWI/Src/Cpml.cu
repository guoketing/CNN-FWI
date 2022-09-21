#include "Model.h"
#include "Cpml.h"
#include "Parameter.h"
#include "utilities.h"


Cpml::Cpml(Parameter &para, Model &model) {

	int nz = model.nz();
	int nx = model.nx();
	int nPml = para.nPoints_pml();
	int nPad = para.nPad();
	float f0 = para.f0();
	float dt = para.dt();
	float dz = para.dz();
	float dx = para.dx();

	float CpAve = compCpAve(model.h_Vp, nz*nx);

	// for padding
	sigma_z = (float*)malloc((nz-nPad)*sizeof(float));
	sigma_x = (float*)malloc(nx*sizeof(float));

	cpmlInit(sigma_z, nz-nPad, nPml, dz, f0, CpAve);
	cpmlInit(sigma_x, nx, nPml, dx, f0, CpAve);

	// for padding
	CHECK(cudaMalloc((void**)&d_sigma_z, (nz-nPad) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_sigma_x, nx *sizeof(float)));

	// for padding
	CHECK(cudaMemcpy(d_sigma_z, sigma_z, (nz-nPad)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_sigma_x, sigma_x, nx*sizeof(float), cudaMemcpyHostToDevice));
}


Cpml::~Cpml() {
	free(sigma_z);
	free(sigma_x);

	CHECK(cudaFree(d_sigma_z));
	CHECK(cudaFree(d_sigma_x));
}