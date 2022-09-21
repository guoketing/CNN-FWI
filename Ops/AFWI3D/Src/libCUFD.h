#ifndef LIBCUFD_H
#define LIBCUFD_H

#include <string>
using std::string;
extern "C" void cufd(float *misfit, float *grad_Vp, float *grad_stf, const float *Vp,
                     const float *stf, int calc_id, const int gpu_id, const int group_size,
                     const int *shot_ids, const string para_fname);
#endif