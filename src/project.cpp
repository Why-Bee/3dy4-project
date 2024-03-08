/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Copyright by Nicola Nicolici
Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include "fourier.h"
#include "genfunc.h"
#include "iofunc.h"
#include "logfunc.h"
#include "demod.h"

#include <limits>
//#include <thread>

// CURRENT TASKS:
// Do the Mono implementation
// Algorithm:
// Step 1: Get data from the RF dongle and store it (IQ samples)
// Step 2: Low-pass filter the data with cutoff of 2.4 Msamples/s
// Step 3: Downsample the data to 240 ksamples/s
// Step 4: Demodulate the data using custom arctan demodulator
// Step 5: Low-pass filter the demodulated data with cutoff of 16 kHz
// Step 6: Downsample to 48 ksamples/s
// Step 7: Output this audio data to file
enum class AudioChan {
	Mono,
	Stereo,
	Rbds,
};

enum class Mode {
	Mode0,
	Mode1,
	Mode2,
	Mode3,
};

constexpr float kRfSampleFrequency = 2.4e6;
constexpr float kRfCutoffFrequency = 100e3;
constexpr unsigned short int kRfNumTaps = 151;
constexpr int kRfDecimation = 10;

constexpr float kMonoSampleFrequency = 240e3;
constexpr float kMonoCutoffFrequency = 16e3;
constexpr unsigned short int kMonoNumTaps = 151*2;
constexpr int kMonoDecimation = 5;


int main(int argc, char* argv[])
{
	// AudioChan audio_chan = AudioChan::Mono;
	// Mode mode = Mode::Mode0;
	size_t block_size = 2 * 1024 * kRfDecimation * kMonoDecimation; // 2.4MSamples/s * 30ms
	uint32_t block_count = 0;

	std::vector<float> rf_state_i(kRfNumTaps-1, 0.0);
	std::vector<float> rf_state_q(kRfNumTaps-1, 0.0);

	float demod_state_i = 0.0;
	float demod_state_q = 0.0;	

	std::vector<float> mono_state(kMonoNumTaps-1);

	std::vector<float> raw_bin_data;
	std::vector<float> raw_bin_data_i;
	std::vector<float> raw_bin_data_q;

	std::vector<float> pre_fm_demod_i;
	std::vector<float> pre_fm_demod_q;

	std::vector<float> rf_coeffs;
	std::vector<float> mono_coeffs;

	std::vector<float> demodulated_samples;

	std::vector<float> float_audio_data;
	std::vector<int16_t> s16_audio_data;

	impulseResponseLPF(kRfSampleFrequency, 
					   kRfCutoffFrequency, 
					   kRfNumTaps,
					   rf_coeffs);

	impulseResponseLPF(kMonoSampleFrequency, 
					   kMonoCutoffFrequency, 
					   kMonoNumTaps,
					   mono_coeffs);

	std::cerr << "block size: " << block_size << std::endl;
	// while (getBinData(raw_bin_data, block_size)) {
	for (unsigned int block_id = 0; ;block_id++) {
		//std::vector<float> raw_bin_data(block_size);
		raw_bin_data.clear(); raw_bin_data.resize(block_size);
		readStdinBlockData(block_size, block_id, raw_bin_data);

		if ((std::cin.rdstate()) != 0){
			std::cerr << "End of input steam" << std::endl;
			exit(1);
		}
		std::cerr << "Read block " << block_id << std::endl;

	
		//std::cerr << "block count: " << block_count++ << std::endl;

		raw_bin_data_i.clear(); 
		//raw_bin_data_i.resize(block_size/2);
		raw_bin_data_q.clear(); 
		//raw_bin_data_q.resize(block_size/2); 
		for (size_t i = 0; i < raw_bin_data.size(); i+=2){
			raw_bin_data_i.push_back(raw_bin_data[i]);
			raw_bin_data_q.push_back(raw_bin_data[i+1]);
		}

		//std::cerr << "split i and q" << std::endl;

		convolveFIRdecim(pre_fm_demod_i, 
						 raw_bin_data_i,
						 rf_coeffs, 
						 rf_state_i,
						 kRfDecimation);

		//std::cerr << "Convolved i" << std::endl;
		

		convolveFIRdecim(pre_fm_demod_q, 
						 raw_bin_data_q,
						 rf_coeffs, 
						 rf_state_q,
						 kRfDecimation);

		//std::cerr << "Convolved q" << std::endl;


		// convolveFIRdecimIQ(pre_fm_demod_i, 
		// 				   pre_fm_demod_q, 
		// 				   raw_bin_data, 
		// 				   rf_coeffs, 
		// 				   rf_state_i,
		// 				   rf_state_q, 
		// 				   kRfDecimation);

		fmDemodulator(pre_fm_demod_i, 
					  pre_fm_demod_q, 
					  demod_state_i, 
					  demod_state_q, 
					  demodulated_samples);

		convolveFIRdecim(float_audio_data, 
						 demodulated_samples, 
						 mono_coeffs, 
						 mono_state,
						 kMonoDecimation);
		
		//std::cerr << "mono audio convolved" << std::endl;
		
		s16_audio_data.clear() ; s16_audio_data.resize(float_audio_data.size());
		for (unsigned int k=0; k<float_audio_data.size(); k++) {
			if (std::isnan(float_audio_data[k])) float_audio_data[k] = 0;
			s16_audio_data[k] = static_cast<int16_t>(float_audio_data[k]*(std::numeric_limits<int16_t>::max()+1));
		}

		fwrite(&s16_audio_data[0], sizeof(int16_t), s16_audio_data.size(), stdout);

		// s16_audio_data.clear();
		// for (unsigned int k = 0; k < float_audio_data.size(); k++){
		// 		if (std::isnan(float_audio_data[k])) s16_audio_data.push_back(0);
		// 		else s16_audio_data.push_back(static_cast<short int>(float_audio_data[k]*16384));
		// }
		// fwrite(&s16_audio_data[0], sizeof(int16_t), s16_audio_data.size(), stdout);
	}

    // processing_thread.join();
    // audio_write_thread.join();
	
	return 0;
}
