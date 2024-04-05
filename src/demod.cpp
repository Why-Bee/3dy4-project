/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "demod.h"


/**
 * @todo
 * 
 * Num Multiplications: I.size() * 5
 * Num Accumulations: 
 * */
void fmDemodulator(const std::vector<float>& I, const std::vector<float>& Q, float& prev_I, float& prev_Q, std::vector<float>& fm_demod_samples) {
    fm_demod_samples.resize(I.size());
    float curr_I, curr_Q;
    for(int k = 0; k < (int)I.size(); ++k) {
        curr_I = I[k];
        curr_Q = Q[k];
        float denominator = std::pow(curr_I, 2) + std::pow(curr_Q, 2);
        if (denominator) {
            fm_demod_samples[k] = ((curr_I * (curr_Q - prev_Q)) - (curr_Q * (curr_I - prev_I))) / denominator;
        }
        else {
            fm_demod_samples[k] = 0.0f;
        }
        prev_I = curr_I;
        prev_Q = curr_Q;
    }
}