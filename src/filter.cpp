/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include "logfunc.h"

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

// function to compute the impulse response "h" based on the sinc function
void impulseResponseLPF(float Fs, float Fc, unsigned short int num_taps, std::vector<float> &h, unsigned int upsampling_factor)
{
	// update num_taps and sampling rate to reflect upsampling factor
	num_taps *= upsampling_factor;
	Fs *= upsampling_factor;

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
		h[i] *= pow(std::sin(i*PI/num_taps), 2) * upsampling_factor;
	}
}


// function to compute the impulse response "h" for a band pass filter
void impulseResponseBPF(float Fs, float Fb, float Fe, unsigned short int num_taps, std::vector<float> &h) 
{
	h.clear(); h.resize(num_taps, 0.0);

	float norm_center = ((Fe + Fb) / 2) / (Fs / 2);	// normalized center freq
	float norm_pass = (Fe - Fb) / (Fs / 2);	// normalized pass band

	for (int i = 0; i < num_taps; ++i) {
		if (i == (num_taps - 1) / 2) {
			h[i] = norm_pass;	// avoid division by zero
		} else {
			h[i] = norm_pass * ((std::sin(PI * (norm_pass / 2) * (i - ((num_taps - 1) / 2)))) / 
								(PI * (norm_pass / 2) * (i - ((num_taps - 1) / 2))));
		}
		h[i] = h[i] * std::cos((i - ((num_taps - 1) / 2)) * PI * norm_center);
		h[i] = h[i] * pow(std::sin((i * PI) / num_taps), 2);
	}
}


void convolveFIR2(std::vector<float> &y, std::vector<float> &x, std::vector<float> &h, std::vector<float> &zi, int decimation)
{
	y.clear(); y.resize(x.size()/decimation, 0.0);
    int decim_n;
    for (int n = 0; n < x.size(); n += decimation) {
        decim_n = n/decimation;
        for (int k = 0; k < h.size(); k++){
            if (n - k >= 0) {
                y[decim_n] += x[n - k] * h[k];
            } else {
                y[decim_n] += zi[zi.size() + (n - k)] * h[k];
            }
        }
    }

    for (int i = x.size() - h.size(); i < x.size(); i++){

      zi[i - x.size() + h.size()] = x[i];

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
	std::vector<float> indices;
    for (int n = 0; n < x.size(); n += decimation) {
        decim_n = n/decimation;
        for (int k = 0; k < h.size(); k++){
			if ( (n - k) >= 0 ) {
				y[decim_n] += h[k] * x[n-k];
			}
			else { // n- k < 0 take from state
				y[decim_n] += h[k] * zi[n-k+(zi.size())];
			}
        }
    }

	for (int i = 0; i < zi.size(); i++) {
		zi[i] = (x.size()-1) + decimation*(-(zi.size()-1) + i);;
	}
}

void upsample(std::vector<float> &y,
			  int upsampling_factor) 
{
	if (upsampling_factor == 1) {
		return;
	}

	int original_size = y.size();
	y.resize(original_size * upsampling_factor);

	for (int i = original_size - 1; i >= 0; i--) {
		y[i * upsampling_factor] = y[i];
		for (int j = 1; j < upsampling_factor; j++) {
			y[i * upsampling_factor + j] = 0.00;
		}
	}
}

void downsample(std::vector<float> &y,
			  	int decimation)
{
	if (decimation == 1) {
		return;
	}
	
	int index = 0;

	for (int i = 0; i < y.size(); i+=decimation) {
		y[index] = y[i];
		index++;
	}
	y.resize(y.size() / decimation);
		std::cerr << y.size() << std::endl;
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
		// input_index = ((decimation * n) - phase) / upsampling_factor;
        for (int k = phase; k < h.size(); k += upsampling_factor){
			input_index = static_cast<int>(((n*decimation)-k) / upsampling_factor);
			if ( input_index >= 0 ) {
				y[n] += h[k] * x[input_index];
			} else { // take from state
				y[n] += h[k] * zi[input_index+(zi.size())] ;
			}
			// input_index--;
        }
    }
    for (int i = x.size() - zi.size(); i < x.size(); i++) {
      zi[i - x.size() + zi.size()] = x[i];
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

