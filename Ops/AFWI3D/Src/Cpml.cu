#include "Model.h"
#include "Cpml.h"
#include "Parameter.h"
#include "utilities.h"


Cpml::Cpml(Parameter &para, Model &model) {

	int nz = model.nz();
	int nx = model.nx();
	int ny = model.ny();
	int nPml = para.nPoints_pml();
	float f0 = para.f0();
	float dt = para.dt();
	float dz = para.dz();
	float dx = para.dx();
	float dy = para.dy();

	float CpAve = compCpAve(model.h_Vp, nz*nx*ny);

	// for padding
	sigma_z = (float*)malloc(nz*sizeof(float));
	sigma_x = (float*)malloc(nx*sizeof(float));
	sigma_y = (float*)malloc(ny*sizeof(float));

	cpmlInit(sigma_z, nz, nPml, dz, f0, CpAve);
	cpmlInit(sigma_x, nx, nPml, dx, f0, CpAve);
	cpmlInit(sigma_y, ny, nPml, dx, f0, CpAve);

	// for padding
	CHECK(cudaMalloc((void**)&d_sigma_z, nz *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_sigma_x, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_sigma_y, ny *sizeof(float)));

	// for padding
	CHECK(cudaMemcpy(d_sigma_z, sigma_z, nz*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_sigma_x, sigma_x, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_sigma_y, sigma_y, ny*sizeof(float), cudaMemcpyHostToDevice));
}


Cpml::~Cpml() {
	free(sigma_z);
	free(sigma_x);
	free(sigma_y);

	CHECK(cudaFree(d_sigma_z));
	CHECK(cudaFree(d_sigma_x));
	CHECK(cudaFree(d_sigma_y));
}