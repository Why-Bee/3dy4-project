/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef DY4_DEMOD_H
#define DY4_DEMOD_H

#include <vector>
#include <cmath>

void fmDemodulator(const std::vector<float>& I, 
                   const std::vector<float>& Q, 
                   float& prev_I, 
                   float& prev_Q, 
                   std::vector<float>& fm_demod_samples);

#endif