/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "demod.h"

// function for FM demodulation without arctan
void fm_demodulator(const std::vector<float>& I, const std::vector<float>& Q, float& prev_I, float& prev_Q, std::vector<float>& fm_demod_samples) {
    fm_demod_samples.resize(I.size());
    for(unsigned int k = 0; k < (int)I.size(); ++k) {
        float denominator = std::pow(I[k], 2) + std::pow(Q[k], 2);
        if (denominator) {
            fm_demod_samples[k] = ((I[k] * (Q[k] - prev_Q)) - (Q[k] * (I[k] - prev_I))) / denominator;
        }
        else {
            fm_demod_samples[k] = 0.0f;
        }
        prev_I = I[k];
        prev_Q = Q[k];
    }
}