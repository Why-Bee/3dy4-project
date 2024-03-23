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
#include "pll.h"

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

constexpr float kRfSampleFrequency = 2.4e6;
constexpr float kRfCutoffFrequency = 100e3;
constexpr unsigned short int kRfNumTaps = 101; // NOTE script works only for 151 here but not smth to rely on
constexpr int kRfDecimation = 10;

constexpr float kMonoSampleFrequency = 240e3;
constexpr float kMonoCutoffFrequency = 16e3;
constexpr unsigned short int kMonoNumTaps = 101; // NOTE script works when I use 151*2 here
constexpr int kMonoDecimation = 5;

constexpr float kStereoBpfFcHigh = 54e3;
constexpr float kStereoBpfFcLow = 22e3;
constexpr float kStereoBpfNumTaps = 101;
constexpr float kStereoDecimation = kMonoDecimation;

constexpr float kStereoLpfFc = 38e3;
constexpr float kStereoLpfNumTaps = 101;
constexpr float kMixerGain = 2.0;

constexpr float kPilotToneFrequency = 19e3;
constexpr float kPilotNcoScale = 2.0;
constexpr float kPilotBpfFcHigh = 19.5e3;
constexpr float kPilotBpfFcLow = 18.5e3;
constexpr float kPilotBpfNumTaps = 101;

constexpr uint16_t kMaxUint14 = 0x3FFF;

constexpr float kIQfactor = 2.0;

#define DEBUG_MODE 0U

int main(int argc, char* argv[])
{
	static constexpr size_t block_size = kIQfactor * 1024 * kRfDecimation * kMonoDecimation;

	std::vector<float> rf_state_i(kRfNumTaps-1, 0.0);
	std::vector<float> rf_state_q(kRfNumTaps-1, 0.0);
	float demod_state_i = 0.0;
	float demod_state_q = 0.0;

	std::vector<float> mono_lpf_state(kMonoNumTaps-1, 0.0);
	std::vector<float> apf_state(static_cast<int>((kStereoLpfNumTaps-1)/2), 0.0);

	std::vector<float> pilot_bpf_state(kPilotBpfNumTaps-1, 0.0);

	std::vector<float> stereo_bpf_state(kStereoBpfNumTaps-1, 0.0);
	std::vector<float> stereo_lpf_state(kStereoLpfNumTaps-1, 0.0);

	PllState pll_state = PllState();

	std::vector<float> raw_bin_data(block_size);
	std::vector<float> raw_bin_data_i;
	std::vector<float> raw_bin_data_q;

	std::vector<float> pre_fm_demod_i;
	std::vector<float> pre_fm_demod_q;

	std::vector<float> rf_coeffs;
	std::vector<float> mono_lpf_coeffs;
	std::vector<float> stereo_bpf_coeffs;
	std::vector<float> stereo_lpf_coeffs;
	std::vector<float> pilot_bpf_coeffs;

	std::vector<float> demodulated_samples;
	std::vector<float> demodulated_samples_delayed(block_size/(kIQfactor*kRfDecimation), 0.0);

	std::vector<float> pilot_filtered(block_size/(kIQfactor*kRfDecimation), 0.0);
	std::vector<float> stereo_bpf_filtered(block_size/(kIQfactor*kRfDecimation), 0.0);
	std::vector<float> stereo_mixed(block_size/(kIQfactor*kRfDecimation), 0.0);
	std::vector<float> nco_out; // block_size/kRfDecimation + 1
	std::vector<float> stereo_lpf_filtered(block_size/(kIQfactor*kRfDecimation*kStereoDecimation), 0.0);

	std::vector<float> float_mono_data;

	std::vector<float> float_stereo_left_data(block_size/(kIQfactor*kRfDecimation*kStereoDecimation), 0.0);
	std::vector<float> float_stereo_right_data(block_size/(kIQfactor*kRfDecimation*kStereoDecimation), 0.0);

	std::vector<short int> s16_audio_data(float_stereo_right_data.size()*2);

	std::vector<float> nco_out_debug;
	std::vector<float> pilot_debug;

	/* Parse command line arguments */
	int mode = 0;
	int channel = 0;

	if (argc < 2) {
		std::cerr << "Operating in default mode 0 and channel 0 (mono)" << std::endl;
	} else if (argc == 2 || argc == 3) {
		mode = std::atoi(argv[1]);
		if (mode > 3 || mode < 0) {
			std::cerr << "Invalid mode entered: " << mode << std::endl;
			exit(1);
		}
		if (argc == 3) {
			channel = std::atoi(argv[2]);
			if (channel > 1 || channel < 0) {
				std::cerr << "Invalid channel entered: " << channel << std::endl;
				exit(1);
			}
		}
	} else {
		std::cerr << "Usage: " << argv[0] << std::endl;
		std::cerr << "or " << argv[0] << " <mode>" << std::endl;
		std::cerr << "\t\t <mode> is a value from 0 to 3" << std::endl;
		exit(1);
	}

	if (channel == 0) {
		std::cerr << "Operating in mode " << mode << " with mono channel" << std::endl;
	} else if (channel == 1) {
		std::cerr << "Operating in mode " << mode << " with stereo channel" << std::endl;
	}

	impulseResponseLPF(kRfSampleFrequency, 
					   kRfCutoffFrequency, 
					   kRfNumTaps,
					   rf_coeffs);

	logVector("impulse_resp_rf", rf_coeffs);

	impulseResponseLPF(kMonoSampleFrequency, 
					   kMonoCutoffFrequency, 
					   kMonoNumTaps,
					   mono_lpf_coeffs);

	logVector("impulse_resp_mono", mono_lpf_coeffs);

	impulseResponseBPF(kMonoSampleFrequency,
					kStereoBpfFcLow,
					kStereoBpfFcHigh,
					kStereoBpfNumTaps,
					stereo_bpf_coeffs);

	logVector("impulse_resp_stereo_bpf", stereo_bpf_coeffs);

	impulseResponseLPF(kMonoSampleFrequency,
					   kStereoLpfFc,
					   kStereoLpfNumTaps,
					   stereo_lpf_coeffs);

	logVector("impulse_resp_stereo_lpf", stereo_lpf_coeffs);

	impulseResponseBPF(kMonoSampleFrequency,
					   kPilotBpfFcLow,
					   kPilotBpfFcHigh,
					   kPilotBpfNumTaps,
					   pilot_bpf_coeffs);

	raw_bin_data_i.clear(); raw_bin_data_i.resize(block_size/2);
	raw_bin_data_q.clear(); raw_bin_data_q.resize(block_size/2);

	std::cerr << "block size: " << block_size << std::endl;
	for (unsigned int block_id = 0; ;block_id++) {
		readStdinBlockData(block_size, block_id, raw_bin_data);

		if ((std::cin.rdstate()) != 0){
			std::cerr << "End of input stream reached" << std::endl;
			exit(1);
		}

		std::cerr << "Read block " << block_id << std::endl;

		for (size_t i = 0; i < raw_bin_data.size(); i+=2){
			raw_bin_data_i[i>>1] = raw_bin_data[i];
			raw_bin_data_q[i>>1] = raw_bin_data[i+1];
		}

		convolveFIR2(pre_fm_demod_i, 
					 raw_bin_data_i,
					 rf_coeffs, 
					 rf_state_i,
					 kRfDecimation);

		convolveFIR2(pre_fm_demod_q, 
					 raw_bin_data_q,
					 rf_coeffs, 
					 rf_state_q,
					 kRfDecimation);

		fmDemodulator(pre_fm_demod_i, 
					  pre_fm_demod_q, 
					  demod_state_i, 
					  demod_state_q, 
					  demodulated_samples);
		
		delayBlock(demodulated_samples,
				   demodulated_samples_delayed,
				   apf_state);

		convolveFIR2(float_mono_data, 
					demodulated_samples_delayed,
					mono_lpf_coeffs, 
					mono_lpf_state,
					kMonoDecimation);	

		convolveFIR(stereo_bpf_filtered,
					demodulated_samples,
					stereo_bpf_coeffs,
					stereo_bpf_state);
		
		convolveFIR(pilot_filtered,
					demodulated_samples,
					pilot_bpf_coeffs,
					pilot_bpf_state);

		fmPll(pilot_filtered,
			  kPilotToneFrequency,
			  kMonoSampleFrequency,
			  pll_state,
			  kPilotNcoScale,
			  nco_out);
		
		// Mixer @copyright Samuel Parent
		for (size_t i = 0; i < stereo_bpf_filtered.size(); i++) {
			stereo_mixed[i] = kMixerGain*nco_out[i]*stereo_bpf_filtered[i];
		}

		convolveFIR2(stereo_lpf_filtered,
					 stereo_mixed,
					 stereo_lpf_coeffs,
					 stereo_lpf_state,
					 kStereoDecimation);

		for (size_t i = 0; i < stereo_lpf_filtered.size(); i++) {
			float_stereo_left_data[i] = float_mono_data[i] + stereo_lpf_filtered[i];
			float_stereo_right_data[i] = float_mono_data[i] - stereo_lpf_filtered[i];
		}

		for (unsigned int k = 0; k < float_stereo_right_data.size(); k++){
				if (std::isnan(float_stereo_right_data[k]) || std::isnan(float_stereo_left_data[k])) {
					s16_audio_data[2*k] = 0;
					s16_audio_data[2*k + 1] = 0;
				} else {
					s16_audio_data[2*k] = static_cast<short int>(float_stereo_right_data[k]*(kMaxUint14+1));
					s16_audio_data[2*k + 1] = static_cast<short int>(float_stereo_left_data[k]*(kMaxUint14+1));
				}
		}
		fwrite(&s16_audio_data[0], sizeof(short int), s16_audio_data.size(), stdout);

		#if DEBUG_MODE
		if (block_id == 100 || block_id == 1) {
			std::cerr << "Logging Vectors" << std::endl;
		 	logVector("demodulated_samples" + std::to_string(block_id), demodulated_samples);	
			logVector("demodulated_samples_delayed" + std::to_string(block_id), demodulated_samples_delayed);	
			logVector("float_mono_data" + std::to_string(block_id), float_mono_data);
			logVector("pilot_filtered" + std::to_string(block_id), pilot_filtered);
			logVector("nco_out" + std::to_string(block_id), nco_out);
			logVector("stereo_mixed" + std::to_string(block_id), stereo_mixed);
			logVector("stereo_lpf_filtered" + std::to_string(block_id), stereo_lpf_filtered);
			logVector("float_stereo_left_data" + std::to_string(block_id), float_stereo_left_data);
			logVector("float_stereo_right_data" + std::to_string(block_id), float_stereo_right_data);
		}

		if (block_id == 227 || block_id == 228) {
			for (int i = 0; i < pilot_filtered.size(); i++) {
				nco_out_debug.push_back(nco_out[i]);
				pilot_debug.push_back(pilot_filtered[i]);
			}
		} else if (block_id == 229) {
			logVector("nco_out_debug", nco_out_debug);
			logVector("pilot_filtered_debug", pilot_debug);
		}
		#endif 
	}
	
	return 0;
}
