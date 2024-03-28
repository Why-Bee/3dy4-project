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
#include "rds.h"

#include <limits>
#include <unordered_map>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <pthread.h>
#include <string>

void rf_frontend_thread(SafeQueue<std::vector<float>> &demodulated_samples_queue);
void audio_processing_thread(SafeQueue<std::vector<float>> &demodulated_samples_queue);
void rds_processing_thread(SafeQueue<std::vector<float>> &demodulated_samples_queue);

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

constexpr float kRDSBpfFcHigh = 60e3;
constexpr float kRDSBpfFcLow = 54e3;
constexpr float kRDSBpfNumTaps = 101;
constexpr float kRDSLpfFc = 3e3;
constexpr float kRDSLpfNumTaps = 101;
constexpr float kRDSRrcNumTaps = 151;
constexpr float kRDSCarrierFreq = 114e3;
constexpr float kRDSSquaredBpfFcHigh = 114.5e3;
constexpr float kRDSSquaredBpfFcLow = 113.5e3;
constexpr float kRDSSquaredBpfNumTaps = 101;
constexpr float kRDSNcoScale = 0.5;

constexpr uint16_t kMaxUint14 = 0x3FFF;

#define DEBUG_MODE 0U

/* GLOBAL VARIABLES */

std::unordered_map<uint8_t, Config> config_map = {
	{0, {.block_size= 76800, .rf_downsample=10, AudioConfig{.upsample=  1, .downsample=   5}, RdsConfig{.upsample=247, .downsample=1920, .sps=13, .spb_aggr=76}}},
	{1, {.block_size= 86400, .rf_downsample= 6, AudioConfig{.upsample=  1, .downsample=  12}, RdsConfig{.upsample=  0, .downsample=   0, .sps= 0, .spb_aggr=0}}},
	{2, {.block_size= 96000, .rf_downsample=10, AudioConfig{.upsample=147, .downsample= 800}, RdsConfig{.upsample= 19, .downsample=  64, .sps=30, .spb_aggr=95}}},
	{3, {.block_size=115200, .rf_downsample= 9, AudioConfig{.upsample=441, .downsample=3200}, RdsConfig{.upsample=  0, .downsample=   0, .sps= 0, .spb_aggr=0}}},
};

int mode    = 0; // mode 0, 1, 2, 3. Default is 0
int channel = 0; // 1 for mono, 2 for stereo (rds runs in stereo)

int main(int argc, char* argv[])
{
	argparse_mode_channel(argc, argv, mode, channel);

	// Queue for demodulated samples shared between threads
	SafeQueue<std::vector<float>> demodulated_samples_queue;

	std::thread rf_processing_thread(rf_frontend_thread, std::ref(demodulated_samples_queue));
	std::thread audio_consumer_thread(rds_processing_thread, std::ref(demodulated_samples_queue));
	// std::thread audio_consumer_thread(audio_processing_thread, std::ref(demodulated_samples_queue));

	#ifndef __APPLE__
	cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);
    pthread_setaffinity_np(rf_processing_thread.native_handle(), 
						   sizeof(cpu_set_t), 
						   &cpuset);

	CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);
	pthread_setaffinity_np(audio_consumer_thread.native_handle(), 
						   sizeof(cpu_set_t), 
						   &cpuset);
	#endif

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
		// auto cpu_id = sched_getcpu();
		std::cerr << "Read block " << block_id << std::endl;

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
	std::vector<float> stereo_lpf_filtered(block_size*audio_upsample/(kIQfactor*rf_decimation*audio_decimation), 0.0);

	std::vector<float> float_mono_data;

	std::vector<float> float_stereo_left_data(block_size*audio_upsample/(kIQfactor*rf_decimation*audio_decimation), 0.0);
	std::vector<float> float_stereo_right_data(block_size*audio_upsample/(kIQfactor*rf_decimation*audio_decimation), 0.0);


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

			convolveFIRResample(float_mono_data, 
						demodulated_samples_delayed,
						mono_coeffs, 
						mono_lpf_state,
						audio_decimation,
						audio_upsample);	

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
			
			for (size_t i = 0; i < stereo_bpf_filtered.size(); i++) {
				stereo_mixed[i] = kMixerGain*nco_out[i]*stereo_bpf_filtered[i];
			}

			convolveFIRResample(stereo_lpf_filtered,
						stereo_mixed,
						stereo_lpf_coeffs,
						stereo_lpf_state,
						audio_decimation,
						audio_upsample);

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

		// auto cpu_id = sched_getcpu();
		// std::cerr << "Audio CPU: " << cpu_id << std::endl;

	}
}

void rds_processing_thread(SafeQueue<std::vector<float>> &demodulated_samples_queue)
{
	static const size_t block_size = config_map.at(mode).block_size;
	static const short int rf_decim = config_map.at(mode).rf_downsample;
	static const short int rds_upsample = config_map.at(mode).rds.upsample;
	static const short int rds_downsample = config_map.at(mode).rds.downsample;
	static const short int rds_sps = config_map.at(mode).rds.sps;
	static const short int rds_spb_aggr = config_map.at(mode).rds.spb_aggr;

	constexpr int post_rrc_filt_aggr_blocks = 2; // DONT TRY TO UPDATE THIS UNLESS CHANGING CONFIG for SPB
	constexpr int post_samp_pts_aggr_blocks = 2;

	float rds_decim = rds_downsample/rds_upsample;

	std::vector<float> rds_bpf_coeffs;
	std::vector<float> rds_lpf_coeffs;
	std::vector<float> rds_squared_bpf_coeffs;
	std::vector<float> rds_rrc_coeffs;

	std::vector<float> rds_bpf_state(kRDSBpfNumTaps-1, 0.0);
	std::vector<float> rds_lpf_state((kRDSLpfNumTaps-1), 0.0); 
	std::vector<float> rds_apf_state(static_cast<int>((kRDSBpfNumTaps-1)/2), 0);
	std::vector<float> rds_squared_bpf_state(kRDSSquaredBpfNumTaps-1, 0.0);
	std::vector<float> rds_rrc_state(kRDSRrcNumTaps-1, 0.0);

	PllState pll_state_rds = PllState();

	std::vector<float> rds_filtered(block_size/(kIQfactor*rf_decim),0.0);
	std::vector<float> rds_squared(block_size/(kIQfactor*rf_decim),0.0);
	std::vector<float> rds_pilot(block_size/(kIQfactor*rf_decim),0.0);
	std::vector<float> nco_rds_out(block_size/(kIQfactor*rf_decim)+1, 0.0);
	std::vector<float> rds_delayed(block_size/(kIQfactor*rf_decim), 0.0);
	std::vector<float> rds_mixed(block_size/(kIQfactor*rf_decim), 0.0);
	std::vector<float> rds_mixed_lfiltered(block_size*rds_upsample/(kIQfactor*rf_decim*rds_downsample), 0.0);
	std::vector<float> rds_rrc_filt(block_size*rds_upsample/(kIQfactor*rf_decim*rds_downsample), 0.0); //TODO: scale this with the rational resampled size
	std::vector<float> rds_rrc_filt_aggr(post_rrc_filt_aggr_blocks*block_size*rds_upsample/(kIQfactor*rf_decim*rds_downsample), 0.0); //TODO: scale this with the rational resampled size

	std::vector<float> sampling_points(rds_spb_aggr, 0.0);
	std::vector<float> sampling_points_aggr(post_samp_pts_aggr_blocks*rds_spb_aggr, 0.0);

	std::vector<bool> bitstream(post_samp_pts_aggr_blocks*rds_spb_aggr/2, 0.0);
	std::vector<bool> bitstream_decoded(post_samp_pts_aggr_blocks*rds_spb_aggr/2, 0.0);

	std::vector<bool> fs_state_values(kCheckLen, false);
	
	bool diff_decode_state = 0;
	int post_rrc_filt_block_aggr_counter = 0;
	int post_sample_block_aggr_counter = 0;

	int sampling_start_offset = 0;
	int num_blocks_for_pll_tuning = 20;
	int num_blocks_for_cdr = 10;
	int num_blocks_for_cdr_counter = 0;

	int bitstream_select = -1;
	int bitstream_score_0 = 0;
	int bitstream_score_1 = 0;
	int bitstream_select_thresh = 5;
	bool recov_bistream_state = false;

	int fs_found_count = 0;
	int fs_last_found_counter = 0;
	char fs_expected_next = '\0';
	int fs_state_len = 0;
	int fs_mode = 1;
	int fs_init_found_thresh = 4;
	const int fs_rubish_tresh = 10;

	uint16_t fs_rubish_score = 0;
	uint16_t fs_rubish_streak = 0;
	uint32_t ps_next_up = -1;
	uint32_t ps_next_up_pos = 0;
	uint8_t ps_num_chars_set = 0;
	std::string program_service = "________";

	impulseResponseBPF(kMonoSampleFrequency,
					   kRDSBpfFcLow,
					   kRDSBpfFcHigh,
					   kRDSBpfNumTaps,
					   rds_bpf_coeffs);

	impulseResponseBPF(kMonoSampleFrequency,
					   kRDSSquaredBpfFcLow,
					   kRDSSquaredBpfFcHigh,
					   kRDSSquaredBpfNumTaps,
					   rds_squared_bpf_coeffs);

	impulseResponseLPF(kMonoSampleFrequency,
					   kRDSLpfFc,
					   kRDSLpfNumTaps,
					   rds_lpf_coeffs,
					   rds_upsample);
	
	impulseResponseRRC(kMonoSampleFrequency/rds_decim,
					   kRDSRrcNumTaps,
					   rds_rrc_coeffs);

	for (uint64_t block_count = 0; ;block_count++) 
	{
		std::vector<float> fm_demodulated = demodulated_samples_queue.dequeue();

		convolveFIR(rds_filtered, fm_demodulated, rds_bpf_coeffs, rds_bpf_state); // get the entire RDS data

		for (unsigned int i = 0; i < rds_filtered.size(); i++)
			rds_squared[i] = rds_filtered[i]*rds_filtered[i];

		convolveFIR(rds_pilot, rds_squared, rds_squared_bpf_coeffs, rds_squared_bpf_state); // extract the carrier

		fmPll(rds_pilot, kRDSCarrierFreq, kMonoSampleFrequency, pll_state_rds, kRDSNcoScale, nco_rds_out, 0, 0.0005); // lock pll at 57 kHz to pilot

		if (block_count < num_blocks_for_pll_tuning)
			continue;

		// pll is tuned now

		delayBlock(rds_filtered, rds_delayed, rds_apf_state); // delay the rds to match filtering on carrier

		// DEBUG: delete later if needed
		if (nco_rds_out.size()-1 != rds_delayed.size())
			std::cerr << "WARNING- sizing error on RDS path! NCO size: " << nco_rds_out.size() << " RDS data size: " << rds_delayed.size() << std::endl;

		// Mixer!! credit Yash Bhatia, bhatiy1@mcmaster.ca, very cool guy, contact for licensing fees
		for (unsigned int i = 0; i < rds_delayed.size(); i++)
			rds_mixed[i] = 2*rds_delayed[i]*nco_rds_out[i];

		convolveFIRResample(rds_mixed_lfiltered, rds_mixed, rds_lpf_coeffs, rds_lpf_state, rds_downsample, rds_upsample); // Filter to 3kHz
		// TODO: this is not known good, need to test?

		convolveFIR(rds_rrc_filt, rds_mixed_lfiltered, rds_rrc_coeffs, rds_rrc_state); // Convert to a Root Raised Cosine

		// RRC wave established- time to recover data

		if (block_count == 120) {
			std::cerr << "LOGGGGGGGGG" << std::endl;
			logVector("rds_pll_out", nco_rds_out);
			logVector("rds_data_apf", rds_delayed);
			logVector("rds_pilot", rds_pilot);
			logVector("rds_rrc_filt", rds_rrc_filt);
		}

		// Aggregate 2 blocks together before sampling
		if (post_rrc_filt_block_aggr_counter == 0) {
			for (int i = 0; i < rds_rrc_filt.size(); i++) {
				rds_rrc_filt_aggr[i] = rds_rrc_filt[i];
			}
		} else if (post_rrc_filt_block_aggr_counter > 0) {
			int offset = rds_rrc_filt.size()*post_rrc_filt_block_aggr_counter;
			for (int i = 0; i < rds_rrc_filt.size(); i++) {
				rds_rrc_filt_aggr[offset + i] = rds_rrc_filt[i];
			}
		}

		if (post_rrc_filt_block_aggr_counter<(post_rrc_filt_aggr_blocks-1)) {
			post_rrc_filt_block_aggr_counter++;
			continue;
		}

		post_rrc_filt_block_aggr_counter = 0;

		if (num_blocks_for_cdr_counter < num_blocks_for_cdr) {
			sampling_start_offset += sampling_start_adjust(rds_rrc_filt_aggr, rds_sps);
			num_blocks_for_cdr_counter++;
			continue;
		} else if (num_blocks_for_cdr_counter == num_blocks_for_cdr) {
			num_blocks_for_cdr_counter ++;
			sampling_start_offset = static_cast<int>(sampling_start_offset/num_blocks_for_cdr);
			std::cerr << "sampling start offset " << sampling_start_offset << std::endl; 
		}

		for (int i = sampling_start_offset, j = 0; i < rds_rrc_filt_aggr.size(); i+=rds_sps, j++) {
			sampling_points[j] = rds_rrc_filt_aggr[i];
		}

		if (post_sample_block_aggr_counter == 0) {
			for (int i = 0; i < size(sampling_points); i++) {
				sampling_points_aggr[i] = sampling_points[i];
			}
		} else if (post_sample_block_aggr_counter > 0) {
			size_t offset = sampling_points.size()*post_sample_block_aggr_counter;
			for (unsigned int i = 0; i < sampling_points.size(); i++) {
				sampling_points_aggr[offset + i] = sampling_points[i];
			}
		}

		if (post_sample_block_aggr_counter < (post_samp_pts_aggr_blocks-1)) {
			post_sample_block_aggr_counter++;
			continue;
		}

		post_sample_block_aggr_counter = 0;

		// the aggregating of data is done

		recover_bitstream(bitstream, 
						  bitstream_select, 
						  bitstream_score_0, 
						  bitstream_score_1,
						  recov_bistream_state,
						  sampling_points_aggr,
						  bitstream_select_thresh);
		
		differential_decode_stateful(bitstream_decoded, diff_decode_state, bitstream);

		if (fs_mode == 1) {
			frame_sync_initial(bitstream_decoded, 
								fs_found_count, 
								fs_last_found_counter, 
								fs_expected_next, 
								fs_state_values, 
								fs_state_len);
			if (fs_found_count >= fs_init_found_thresh) {
				for (int i = 0; i< fs_last_found_counter; i++) {
					fs_state_values[i] = fs_state_values[fs_state_values.size()-fs_last_found_counter+i];
				}
				fs_state_len = fs_last_found_counter;
				fs_mode = 2;
				std::cerr << "MODE 2!!!" << std::endl;
			}
		} else if (fs_mode == 2) {
			frame_sync_blockwise(bitstream_decoded, 
								 fs_expected_next,
								 fs_rubish_score,
								 fs_rubish_streak,
								 fs_state_values,
								 fs_state_len,
								 ps_next_up,
								 ps_next_up_pos, 
								 ps_num_chars_set,
								 program_service);

			if (fs_rubish_streak > fs_rubish_tresh) {
				fs_mode = 1;
				fs_found_count = 0;
				fs_last_found_counter = 0;
				fs_expected_next = '\0';
				fs_state_values.clear(); fs_state_values.resize(kCheckLen);
				fs_state_len = 0;
				fs_rubish_streak = 0;
				fs_rubish_score = 0;
			}

		}
	
	}
}

