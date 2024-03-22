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
void downsample(std::vector<float> &y, int decimation);
void upsample(std::vector<float> &y, int upsampling_factor);
void convolveFIR(std::vector<float> &y, 
				 const std::vector<float> &x, 
				 const std::vector<float> &h, 
				 std::vector<float> &zi);
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
void convolveFIR2(std::vector<float> &y, 
				  std::vector<float> &x, 
				  std::vector<float> &h, 
				  std::vector<float> &zi, 
				  int decimation);
void delayBlock(const std::vector<float>& x, 
				std::vector<float>& y, 
				std::vector<float>& state);
#endif // DY4_FILTER_H
