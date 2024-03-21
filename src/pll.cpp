/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "pll.h"
#include "dy4.h"

void fmPll(const std::vector<float>& pll_in,
           const float freq, 
           const float fs,
           PllState& pll_state,
           const float nco_scale,
           std::vector<float>& nco_out,
           const float phase_adjust,
           const float norm_bandwidth) {

    float Cp = 2.666;
    float Ci = 3.555;
    float Kp = norm_bandwidth * Cp;
    float Ki = norm_bandwidth * norm_bandwidth * Ci;
    float errorI;
    float errorQ;
    float errorD;
    float trigArg;

    nco_out.clear(); nco_out.resize(pll_in.size()+1);

    nco_out[0] = pll_state.lastNco;

    for (size_t k = 0; k < pll_in.size(); k++){
        errorI = pll_in[k]*pll_state.feedbackI;
        errorQ = pll_in[k]*(-1)*pll_state.feedbackQ;

        errorD = atan2f(errorQ, errorI);

        pll_state.integrator = pll_state.integrator + Ki*errorD;

        pll_state.phaseEst = pll_state.phaseEst + Kp*errorD + pll_state.integrator;

        pll_state.trigOffset++;

        trigArg = (2*PI*freq/fs)*pll_state.trigOffset + pll_state.phaseEst;

        pll_state.feedbackI = cosf(trigArg);
        pll_state.feedbackQ = sinf(trigArg);

        nco_out[k+1] = cos(trigArg*nco_scale + phase_adjust); 
    }

    pll_state.lastNco = nco_out[nco_out.size()-1];
}