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

#include <limits>
#include <unordered_map>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

void rf_frontend_thread(std::queue<std::vector<float>> &demodulated_samples_queue,
						std::mutex &queue_mutex,
						std::condition_variable &queue_cv,
						std::atomic<int> &num_blocks_processed_atomic);

void audio_processing_thread(std::queue<std::vector<float>> &demodulated_samples_queue,
							 std::mutex &queue_mutex,
							 std::condition_variable &queue_cv,
							 std::atomic<int> &num_blocks_processed_atomic);

constexpr float kRfSampleFrequency = 2.4e6;
constexpr float kRfCutoffFrequency = 100e3;
constexpr unsigned short int kRfNumTaps = 101; // NOTE script works only for 151 here but not smth to rely on
constexpr int kRfDecimation = 9;

constexpr float kMonoSampleFrequency = 240e3;	// UPDATE
constexpr float kMonoCutoffFrequency = 16e3;
constexpr unsigned short int kMonoNumTaps = 101; // NOTE script works when I use 151*2 here
constexpr int kMonoDecimation = 5;

constexpr uint16_t kMaxUint14 = 0x3FFF;

constexpr int kMaxQueueElements = 3; // TODO adjust as needed

#define DEBUG_MODE 1U

// TODO: Do we like this format?
const std::unordered_map<uint8_t, TsConfig> config_map = {
	{0, {102400, 10, TsMonoConfig{1,      5}, TsStereoConfig{1, 1}}},
	{1, {147456, 6,  TsMonoConfig{1,     12}, TsStereoConfig{1, 1}}},
	{2, {112000, 10, TsMonoConfig{147,  800}, TsStereoConfig{1, 1}}},
	{3, {133746, 9,  TsMonoConfig{441, 3200}, TsStereoConfig{1, 1}}},
};

static int mode = 0;
static int channel = 0;
int main(int argc, char* argv[])
{
	/* Parse command line arguments */

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

	std::queue <std::vector<float>> demodulated_samples_queue;
	std::mutex queue_mutex;
	std::condition_variable queue_cv;
	std::atomic<int> num_blocks_processed_atomic;

	std::thread rf_processing_thread(rf_frontend_thread, 
								  std::ref(demodulated_samples_queue),
								  std::ref(queue_mutex),
								  std::ref(queue_cv),
								  std::ref(num_blocks_processed_atomic));

	std::thread audio_consumer_thread(audio_processing_thread,
								   std::ref(demodulated_samples_queue),
								   std::ref(queue_mutex),
								   std::ref(queue_cv),
								   std::ref(num_blocks_processed_atomic));

	rf_processing_thread.join();
	audio_consumer_thread.join();
	
	return 0;
}

void rf_frontend_thread(std::queue<std::vector<float>> &demodulated_samples_queue,
						std::mutex &queue_mutex,
						std::condition_variable &queue_cv,
						std::atomic<int> &num_blocks_processed_atomic)
{
	static const size_t block_size = config_map.at(mode).block_size;

	std::vector<float> rf_state_i(kRfNumTaps-1, 0.0);
	std::vector<float> rf_state_q(kRfNumTaps-1, 0.0);
	
	float demod_state_i = 0.0;
	float demod_state_q = 0.0;	

	// std::vector<float> raw_bin_data(block_size);
	std::vector<float> raw_bin_data_i;
	std::vector<float> raw_bin_data_q;

	std::vector<float> pre_fm_demod_i;
	std::vector<float> pre_fm_demod_q;

	std::vector<float> rf_coeffs;

	std::vector<float> demodulated_samples;
	std::vector<float> raw_bin_data(block_size);

	impulseResponseLPF(kRfSampleFrequency, 
					   kRfCutoffFrequency, 
					   kRfNumTaps,
					   rf_coeffs);

	raw_bin_data_i.clear(); raw_bin_data_i.resize(block_size/2);
	raw_bin_data_q.clear(); raw_bin_data_q.resize(block_size/2);

	num_blocks_processed_atomic = 0;

	std::cerr << "block size: " << block_size << std::endl;
	for (unsigned int block_id = 0; ;block_id++) {
		readStdinBlockData(block_size, block_id, raw_bin_data);

		if ((std::cin.rdstate()) != 0){
			std::cerr << "End of input stream reached" << std::endl;
			exit(1);
		}

		std::cerr << "Read block " << block_id << std::endl;

		// DO NOT RESIZE THESE
		for (size_t i = 0; i < raw_bin_data.size(); i+=2){
			raw_bin_data_i[i>>1] = raw_bin_data[i];
			raw_bin_data_q[i>>1] = raw_bin_data[i+1];
		}

		convolveFIR2(pre_fm_demod_i, 
					 raw_bin_data_i,
					 rf_coeffs, 
					 rf_state_i,
					 config_map.at(mode).rf_downsample);

		convolveFIR2(pre_fm_demod_q, 
					 raw_bin_data_q,
					 rf_coeffs, 
					 rf_state_q,
					 config_map.at(mode).rf_downsample);

		fmDemodulator(pre_fm_demod_i, 
					  pre_fm_demod_q, 
					  demod_state_i, 
					  demod_state_q, 
					  demodulated_samples);
	}

	std::unique_lock<std::mutex> lock(queue_mutex);
	// wait for the queue to be empty
	while (demodulated_samples_queue.size() >= kMaxQueueElements || num_blocks_processed_atomic) {
		queue_cv.wait(lock);
	}

	demodulated_samples_queue.push(demodulated_samples); // TODO consider pushing reference instead
	num_blocks_processed_atomic++;
	queue_cv.notify_all();
	lock.unlock();
}

void audio_processing_thread(std::queue<std::vector<float>> &demodulated_samples_queue,
							 std::mutex &queue_mutex,
							 std::condition_variable &queue_cv,
							 std::atomic<int> &num_blocks_processed_atomic)
{
	std::vector<float> mono_coeffs;
	std::vector<float> mono_state(kMonoNumTaps-1, 0.0);
	std::vector<float> float_audio_data;
	std::vector<float> demodulated_samples;

	impulseResponseLPF(kMonoSampleFrequency, 
					   kMonoCutoffFrequency, 
					   kMonoNumTaps,
					   mono_coeffs,
					   config_map.at(mode).mono.mono_upsample);
					   
	while (1) {
		std::unique_lock <std::mutex> lock(queue_mutex);
		while (demodulated_samples_queue.empty() || num_blocks_processed_atomic != 1) {
			queue_cv.wait(lock);
		}
		demodulated_samples = demodulated_samples_queue.front();
		demodulated_samples_queue.pop(); // TODO just read here and pop in rds thread
		num_blocks_processed_atomic = 0; // TODO consider incrementing here instead then reset when rds has popped it
		queue_cv.notify_all();
		lock.unlock();
	
		convolveFIRResample(float_audio_data,
							demodulated_samples,
							mono_coeffs,
							mono_state,
							config_map.at(mode).mono.mono_downsample,
							config_map.at(mode).mono.mono_upsample);		 

		std::vector<short int> s16_audio_data(float_audio_data.size());
		for (unsigned int k = 0; k < float_audio_data.size(); k++) {
				if (std::isnan(float_audio_data[k])) s16_audio_data[k] = 0;
				else s16_audio_data[k] = static_cast<short int>(float_audio_data[k]*(kMaxUint14+1));
		}
		fwrite(&s16_audio_data[0], sizeof(short int), s16_audio_data.size(), stdout);
	}
}


	
