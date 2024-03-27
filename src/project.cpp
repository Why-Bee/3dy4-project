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
#include "config.h"
#include "safequeue.h"
#include "pll.h"

#include <limits>
#include <unordered_map>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <pthread.h>

void rf_frontend_thread(SafeQueue<std::vector<float>> &demodulated_samples_queue);
void audio_processing_thread(SafeQueue<std::vector<float>> &demodulated_samples_queue);

constexpr float kRfSampleFrequency = 2.4e6;
constexpr float kRfCutoffFrequency = 100e3;
constexpr unsigned short int kRfNumTaps = 101;

constexpr float kMonoSampleFrequency = 240e3;
constexpr float kMonoCutoffFrequency = 16e3;
constexpr unsigned short int kMonoNumTaps = 101;

constexpr float kStereoBpfFcHigh = 54e3;
constexpr float kStereoBpfFcLow = 22e3;
constexpr float kStereoBpfNumTaps = 101;

constexpr float kStereoLpfFc = 38e3;
constexpr float kStereoLpfNumTaps = 101;
constexpr float kMixerGain = 2.0;

constexpr float kPilotToneFrequency = 19e3;
constexpr float kPilotNcoScale = 2.0;
constexpr float kPilotBpfFcHigh = 19.5e3;
constexpr float kPilotBpfFcLow = 18.5e3;
constexpr float kPilotBpfNumTaps = 101;
constexpr float kIQfactor = 2.0;

constexpr uint16_t kMaxUint14 = 0x3FFF;

#define DEBUG_MODE 0U

const std::unordered_map<uint8_t, Config> config_map = {
	{.mode=0, {.block_size=76800, .rf_downsample=10, AudioConfig{.upsample=  1, .downsample=   5}, RdsConfig{.upsample=247, .downsample=1920, .sps=13}}},
	{.mode=1, {.block_size=86400, .rf_downsample= 6, AudioConfig{.upsample=  1, .downsample=  12}, RdsConfig{.upsample=  0, .downsample=   0, .sps= 0}}},
	{.mode=2, {.block_size=96000, .rf_downsample=10, AudioConfig{.upsample=147, .downsample= 800}, RdsConfig{.upsample= 19, .downsample=  64, .sps=30}}},
	{.mode=3, {.block_size=57600, .rf_downsample= 9, AudioConfig{.upsample=441, .downsample=3200}, RdsConfig{.upsample=  0, .downsample=   0, .sps= 0}}},
};

/* GLOBAL VARIABLES */
int mode = 0;    // mode 0, 1, 2, 3. Default is 0
int channel = 0; // 1 for mono, 2 for stereo (rds runs in stereo)

int main(int argc, char* argv[])
{
	argparse_mode_channel(argc, argv, mode, channel);

	// Queue for demodulated samples shared between threads
	SafeQueue<std::vector<float>> demodulated_samples_queue;

	std::thread rf_processing_thread(rf_frontend_thread, std::ref(demodulated_samples_queue));
	cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);
    pthread_setaffinity_np(rf_processing_thread.native_handle(), 
						   sizeof(cpu_set_t), 
						   &cpuset);

	std::thread audio_consumer_thread(audio_processing_thread, std::ref(demodulated_samples_queue));
	CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);
	pthread_setaffinity_np(audio_consumer_thread.native_handle(), 
						   sizeof(cpu_set_t), 
						   &cpuset);

	rf_processing_thread.join();
	audio_consumer_thread.join();
	
	return 0;
}

void rf_frontend_thread(SafeQueue<std::vector<float>> &demodulated_samples_queue)

{
	static const size_t block_size = config_map.at(mode).block_size;
	static const short int rf_decimation = config_map.at(mode).rf_downsample;

	std::vector<float> rf_state_i(kRfNumTaps-1, 0.0);
	std::vector<float> rf_state_q(kRfNumTaps-1, 0.0);
	
	float demod_state_i = 0.0;
	float demod_state_q = 0.0;	

	std::vector<float> raw_bin_data_i;
	std::vector<float> raw_bin_data_q;

	std::vector<float> pre_fm_demod_i;
	std::vector<float> pre_fm_demod_q;

	std::vector<float> rf_coeffs;

	std::vector<float> demodulated_samples;

	impulseResponseLPF(kRfSampleFrequency, 
					   kRfCutoffFrequency, 
					   kRfNumTaps,
					   rf_coeffs);

	raw_bin_data_i.resize(block_size/2);
	raw_bin_data_q.resize(block_size/2);


	std::cerr << "block size: " << block_size << std::endl;
	for (unsigned int block_id = 0; ;block_id++) {
		std::vector<float> raw_bin_data(block_size);
		readStdinBlockData(block_size, block_id, raw_bin_data);

		if ((std::cin.rdstate()) != 0){
			std::cerr << "End of input stream reached" << std::endl;
			exit(1);
		}
		auto cpu_id = sched_getcpu();
		std::cerr << "Read block " << block_id << ", CPU: " << cpu_id << std::endl;

		// DO NOT RESIZE THESE
		for (size_t i = 0; i < raw_bin_data.size(); i+=2){
			raw_bin_data_i[i>>1] = raw_bin_data[i];
			raw_bin_data_q[i>>1] = raw_bin_data[i+1];
		}

		convolveFIR2(pre_fm_demod_i, 
					 raw_bin_data_i,
					 rf_coeffs, 
					 rf_state_i,
					 rf_decimation);

		convolveFIR2(pre_fm_demod_q, 
					 raw_bin_data_q,
					 rf_coeffs, 
					 rf_state_q,
					 rf_decimation);

		fmDemodulator(pre_fm_demod_i, 
					  pre_fm_demod_q, 
					  demod_state_i, 
					  demod_state_q, 
					  demodulated_samples);
	
	demodulated_samples_queue.enqueue(demodulated_samples);
	
	}

}

void audio_processing_thread(SafeQueue<std::vector<float>> &demodulated_samples_queue)
{

	static const size_t block_size = config_map.at(mode).block_size;
	static const short int rf_decimation = config_map.at(mode).rf_downsample;
	static const short int audio_decimation = config_map.at(mode).audio.downsample;
	static const short int audio_upsample = config_map.at(mode).audio.upsample;

	std::vector<float> mono_coeffs;
	std::vector<float> mono_state(kMonoNumTaps-1, 0.0);
	std::vector<float> float_audio_data;

	// Stereo related:
	std::vector<float> mono_lpf_state(kMonoNumTaps-1, 0.0);
	std::vector<float> apf_state(static_cast<int>((kStereoLpfNumTaps-1)/2), 0.0);

	std::vector<float> pilot_bpf_state(kPilotBpfNumTaps-1, 0.0);

	std::vector<float> stereo_bpf_state(kStereoBpfNumTaps-1, 0.0);
	std::vector<float> stereo_lpf_state(kStereoLpfNumTaps-1, 0.0);

	PllState pll_state = PllState();

	std::vector<float> mono_lpf_coeffs;
	std::vector<float> stereo_bpf_coeffs;
	std::vector<float> stereo_lpf_coeffs;
	std::vector<float> pilot_bpf_coeffs;

	std::vector<float> demodulated_samples_delayed(block_size/(kIQfactor*rf_decimation), 0.0);

	std::vector<float> pilot_filtered(block_size/(kIQfactor*rf_decimation), 0.0);
	std::vector<float> stereo_bpf_filtered(block_size/(kIQfactor*rf_decimation), 0.0);
	std::vector<float> stereo_mixed(block_size/(kIQfactor*rf_decimation), 0.0);
	std::vector<float> nco_out; // block_size/rf_decimation + 1
	std::vector<float> stereo_lpf_filtered(block_size/(kIQfactor*rf_decimation*audio_decimation), 0.0);

	std::vector<float> float_mono_data;

	std::vector<float> float_stereo_left_data(block_size/(kIQfactor*rf_decimation*audio_decimation), 0.0);
	std::vector<float> float_stereo_right_data(block_size/(kIQfactor*rf_decimation*audio_decimation), 0.0);


	// using mono coeffs as mono lpf coeffs
	impulseResponseLPF(kMonoSampleFrequency, 
					   kMonoCutoffFrequency, 
					   kMonoNumTaps,
					   mono_coeffs,
					   audio_upsample);

	impulseResponseLPF(kMonoSampleFrequency,
					   kStereoLpfFc,
					   kStereoLpfNumTaps,
					   stereo_lpf_coeffs);	

	impulseResponseBPF(kMonoSampleFrequency,
					kStereoBpfFcLow,
					kStereoBpfFcHigh,
					kStereoBpfNumTaps,
					stereo_bpf_coeffs);
	
	impulseResponseBPF(kMonoSampleFrequency,
					   kPilotBpfFcLow,
					   kPilotBpfFcHigh,
					   kPilotBpfNumTaps,
					   pilot_bpf_coeffs);
		   
	while (1) {
		std::vector<float> demodulated_samples = demodulated_samples_queue.dequeue();
	
		convolveFIRResample(float_audio_data,
							demodulated_samples,
							mono_coeffs,
							mono_state,
							audio_decimation,
							audio_upsample);	
			 
		if (channel == 1) {
			delayBlock(demodulated_samples,
					demodulated_samples_delayed,
					apf_state);

			convolveFIR2(float_mono_data, 
						demodulated_samples_delayed,
						mono_coeffs, 
						mono_lpf_state,
						audio_decimation);	

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
						audio_decimation);

			for (size_t i = 0; i < stereo_lpf_filtered.size(); i++) {
				float_stereo_left_data[i] = float_mono_data[i] + stereo_lpf_filtered[i];
				float_stereo_right_data[i] = float_mono_data[i] - stereo_lpf_filtered[i];
			}
		}
		std::vector<short int> s16_audio_data;

		if (channel == 0) { // write mono data
			s16_audio_data.resize(float_audio_data.size());
			for (unsigned int k = 0; k < float_audio_data.size(); k++) {
					if (std::isnan(float_audio_data[k])) s16_audio_data[k] = 0;
					else s16_audio_data[k] = static_cast<short int>(float_audio_data[k]*(kMaxUint14+1));
			}
		}
		else if (channel == 1) { // write stereo data
			s16_audio_data.resize(float_stereo_right_data.size()*2);
			for (unsigned int k = 0; k < float_stereo_right_data.size(); k++){
				if (std::isnan(float_stereo_right_data[k]) || std::isnan(float_stereo_left_data[k])) {
					s16_audio_data[2*k] = 0;
					s16_audio_data[2*k + 1] = 0;
				} else {
					s16_audio_data[2*k] = static_cast<short int>(float_stereo_right_data[k]*(kMaxUint14+1));
					s16_audio_data[2*k + 1] = static_cast<short int>(float_stereo_left_data[k]*(kMaxUint14+1));
				}
			}
		}

		fwrite(&s16_audio_data[0], sizeof(short int), s16_audio_data.size(), stdout);

		auto cpu_id = sched_getcpu();
		std::cerr << "Audio CPU: " << cpu_id << std::endl;

	}
}

