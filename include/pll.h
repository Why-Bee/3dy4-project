
#pragma once

#include <vector>
#include <cmath>

struct PllState
{
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
           const float phase_adjust = 0.0,
           const float norm_bandwidth = 0.01,
           std::vector<float>& nco_out);
