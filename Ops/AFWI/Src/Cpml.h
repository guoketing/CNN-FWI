// Dongzhuo Li 05/09/2018
#ifndef CPML_H__
#define CPML_H__

#include "Parameter.h"
#include "Model.h"


class Cpml {

 public:

	Cpml(Parameter &para, Model &model);
	Cpml(const Cpml&) = delete;
	Cpml& operator=(const Cpml&) = delete;

	~Cpml();

	float *sigma_z;
	float *sigma_x;
	float *d_sigma_z;
	float *d_sigma_x;


};







#endif