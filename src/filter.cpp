/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"

// function to compute the impulse response "h" based on the sinc function
void impulseResponseLPF(float Fs, float Fc, unsigned short int num_taps, std::vector<float> &h)
{
	// allocate memory for the impulse response
	h.clear(); h.resize(num_taps, 0.0);

	float normf = Fc / (Fs/2); // Normalize cutoff
	for (int i = 0; i < num_taps; ++i) {
		if (i == (num_taps - 1) / 2) {
			h[i] = normf;
		}
		else{

			h[i] = normf * ((std::sin(PI * normf * (i - (num_taps - 1) / 2))) / (PI * normf * (i - (num_taps - 1) / 2) ));	
		}
		h[i] *= pow(std::sin(i*PI/num_taps), 2);
	}
}

// function to compute the filtered output "y" by doing the convolution
// of the input data "x" with the impulse response "h"
void convolveFIR(std::vector<float> &y, const std::vector<float> &x, const std::vector<float> &h, std::vector<float> &zi)
{
	// This function convolves x and h to get y, managing state.
	// allocate memory for the output (filtered) data
	y.clear(); y.resize(x.size(), 0.0);

	for(int n = 0; n < y.size(); ++n) {
		for(int k = 0; k < h.size(); ++k ){
			if ( n - k >= 0 && n - k < x.size() ) {
				y[n] += h[k] * x[n-k];
			}
			else { // n- k < 0 take from state
				y[n] += h[k] * zi[n-k+(zi.size())];
			}
		}
	}
	for (unsigned int i = 0; i < zi.size(); ++i)	{
		zi[i] = x[x.size() - zi.size() + i];
	}
}

void convolveFIRdecim(std::vector<float> &y, 
					  const std::vector<float> &x, 
					  const std::vector<float> &h, 
					  std::vector<float> &zi, 
					  int decimation)
{
	// This function convolves x and h to get y, managing state, and downsamples by decimation
	// TODO consider empty initial zi
	// allocate memory for the output (filtered) data
	y.clear(); y.resize(x.size()/decimation, 0.0);

	int decim_n;
    for (int n = 0; n < x.size(); n += decimation) {
        decim_n = n/decimation;
        for (int k = 0; k < h.size(); k++){
			if ( n - k >= 0 ) {
				y[decim_n] += h[k] * x[n-k];
			}
			else { // n- k < 0 take from state
				y[decim_n] += h[k] * zi[n-k+(zi.size())];
			}
        }
    }
    for (int i = x.size() - h.size(); i < x.size(); i++){
      zi[i - x.size() + h.size()] = x[i];
    }
}

void convolveFIRResample(std::vector<float> &y, 
					  const std::vector<float> &x, 
					  const std::vector<float> &h, 
					  std::vector<float> &zi, 
					  int decimation,
					  int upsampling_factor)
{
	// TODO verify functionality
	y.clear(); y.resize(x.size()*upsampling_factor/decimation, 0.0);

	int phase, input_index;
    for (int n = 0; n < y.size(); n++) {
        phase = (n*decimation)%upsampling_factor;
        for (int k = phase; k < h.size(); k += upsampling_factor){
			input_index = static_cast<int>(n*decimation/upsampling_factor) - (k)/upsampling_factor;
			if ( input_index >= 0 ) {
				y[n] += h[k] * x[input_index] * upsampling_factor;
			} else { // take from state
				y[n] += h[k] * zi[input_index+(zi.size())] * upsampling_factor;
			}
        }
    }
    for (int i = x.size() - zi.size(); i < x.size(); i++){
      zi[i - x.size() + h.size()] = x[i];
    }
}

void convolveFIRdecimIQ(std::vector<float> &i_out, 
					  std::vector<float> &q_out, 
					  const std::vector<float> &x,
					  const std::vector<float> &h,
					  std::vector<float> &state_i,
					  std::vector<float> &state_q,
					  const int decimation)
{
	// This function convolves x and h to get y, managing state, and downsamples by decimation
	// TODO consider empty initial zi
	// allocate memory for the output (filtered) data
	constexpr int kInterleavedFactor = 2;

	i_out.clear(); i_out.resize(x.size()/(decimation*kInterleavedFactor), 0.0);
	q_out.clear(); q_out.resize(x.size()/(decimation*kInterleavedFactor), 0.0);

	std::vector<float>* iq_out[kInterleavedFactor] = {&i_out, &q_out};
	std::vector<float>* iq_state[kInterleavedFactor] = {&state_i, &state_q};

	int decim_n;
	for (unsigned int interleaved_offset = 0; interleaved_offset < kInterleavedFactor; interleaved_offset++) {
        for (unsigned int n = interleaved_offset; n < x.size(); n += decimation * kInterleavedFactor) {
            decim_n = n / decimation;
            for (unsigned int k = 0; k < h.size(); k++) {
                if (n - k >= 0) {
                    (*iq_out[interleaved_offset])[decim_n] += h[k] * x[n - k];
                } else { // n - k < 0 take from state
                    (*iq_out[interleaved_offset])[decim_n] +=
                        h[k] * (*iq_state[interleaved_offset])[n - k + iq_state[interleaved_offset]->size()];
                }
            }
        }
        for (unsigned int i = 0; i < iq_state[interleaved_offset]->size(); ++i) {
            (*iq_state[interleaved_offset])[i] = x[x.size() - iq_state[interleaved_offset]->size() + i];
        }
    }
}

