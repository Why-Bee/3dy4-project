
#include <vector>
#include <cmath>

#pragma once

struct PllState
{
    PllState(): 
        integrator(0.0f),
        phaseEst(0.0f),
        feedbackI(1.0f),
        feedbackQ(0.0f),
        trigOffset(0.0f),
        lastNco(1.0f) {}
    float integrator = 0.0f;
    float phaseEst = 0.0f;
    float feedbackI = 1.0f;
    float feedbackQ = 0.0f;
    float trigOffset = 0.0f;
    float lastNco = 1.0f;
};

void fmPll(const std::vector<float>& pll_in,
           const float freq, 
           const float fs,
           PllState& pll_state,
           const float nco_scale,
           std::vector<float>& nco_out,
           const float phase_adjust = 0.0,
           const float norm_bandwidth = 0.01);
