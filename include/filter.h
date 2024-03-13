/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef DY4_FILTER_H
#define DY4_FILTER_H

// add headers as needed
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cmath>

// declaration of a function prototypes
void impulseResponseLPF(float, float, unsigned short int, std::vector<float> &);
void impulseResponseBPF(float Fs, float Fb, float Fe, unsigned short int num_taps, std::vector<float> &h);
void convolveFIR(std::vector<float> &, const std::vector<float> &, const std::vector<float> &);
void convolveFIRdecimIQ(std::vector<float> &i_out, 
					  std::vector<float> &q_out, 
					  const std::vector<float> &x,
					  const std::vector<float> &h,
					  std::vector<float> &state_i,
					  std::vector<float> &state_q,
					  const int decimation);
void convolveFIRdecim(std::vector<float> &y, 
                      const std::vector<float> &x, 
                      const std::vector<float> &h, 
                      std::vector<float> &zi, 
                      int decimation);
void convolveFIRResample(std::vector<float> &y, 
					  const std::vector<float> &x, 
					  const std::vector<float> &h, 
					  std::vector<float> &zi, 
					  int decimation,
					  int upsampling_factor);
#endif // DY4_FILTER_H
