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
#include "cxxopts.hpp"

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

int main(int argc, char* argv[])
{
	AudioChan audio_chan = AudioChan::Mono;
	Mode mode = Mode::Mode0;

	// cxxopts::Options options("3DY4 Project", "Group 30");

    // options.add_options()
    //     ("c,channel", "Select path [m, s, r]")
    //     ("m,mode", "Select mode [0, 1, 2, 3]");

    // try {
    //     auto result = options.parse(argc, argv);

    //     if (result.count("channel")) {
    //         audio_chan = AudioChan::MONO;
    //     } else {
	// 		std::cerr << "No channel passed, using Mono." << std::endl;
	// 	}

    //     std::cerr << "Output file: " << result["output"].as<std::string>() << std::endl;
    // } catch (const cxxopts::OptionException& e) {
    //     std::cerr << "Error parsing options: " << e.what() << std::endl;
    //     return 1;
    // }
	
	size_t block_size = 10000;
	std::vector<float> bin_data(block_size);
	getBinData(bin_data, block_size);

	// // binary files can be generated through the
	// // Python models from the "../model/" sub-folder
	// const std::string in_fname = "../data/fm_demod_10.bin";
	// std::vector<float> bin_data;
	// readBinData(in_fname, bin_data);

	// generate an index vector to be used by logVector on the X axis
	std::vector<float> vector_index;
	genIndexVector(vector_index, bin_data.size());
	// log time data in the "../data/" subfolder in a file with the following name
	// note: .dat suffix will be added to the log file in the logVector function
	logVector("demod_time", vector_index, bin_data);

	// take a slice of data with a limited number of samples for the Fourier transform
	// note: NFFT constant is actually just the number of points for the
	// Fourier transform - there is no FFT implementation ... yet
	// unless you wish to wait for a very long time, keep NFFT at 1024 or below
	std::vector<float> slice_data = \
		std::vector<float>(bin_data.begin(), bin_data.begin() + NFFT);
	// note: make sure that binary data vector is big enough to take the slice

	// declare a vector of complex values for DFT
	std::vector<std::complex<float>> Xf;
	// ... in-lab ...
	// compute the Fourier transform
	// the function is already provided in fourier.cpp
	DFT(slice_data, Xf);

	// compute the magnitude of each frequency bin
	// note: we are concerned only with the magnitude of the frequency bin
	// (there is NO logging of the phase response)
	std::vector<float> Xmag;
	// ... in-lab ...
	// compute the magnitude of each frequency bin
	// the function is already provided in fourier.cpp
	computeVectorMagnitude(Xf, Xmag);

	// log the frequency magnitude vector
	vector_index.clear();
	genIndexVector(vector_index, Xmag.size());
	logVector("demod_freq", vector_index, Xmag); // log only positive freq

	// for your take-home exercise - repeat the above after implementing
	// your OWN function for PSD based on the Python code that has been provided
	// note the estimate PSD function should use the entire block of "bin_data"
	//
	// ... complete as part of the take-home ...
	//
	std::vector<float> freq, psd_est;
	const float Fs = 240;
	estimatePSD(freq, psd_est, bin_data, Fs);
	logVector("demod_psd", freq, psd_est);

	// if you wish to write some binary files, see below example
	//
	// const std::string out_fname = "../data/outdata.bin";
	// writeBinData(out_fname, bin_data);
	//
	// output files can be imported, for example, in Python
	// for additional analysis or alternative forms of visualization

	// nayturally, you can comment the line below once you are comfortable to run GNU plot
	std::cerr << "Run: gnuplot -e 'set terminal png size 1024,768' ../data/example.gnuplot > ../data/example.png\n";
	std::string commandLine= "gnuplot -e 'set terminal png size 1024,768' ../data/example.gnuplot > ../data/example.png";
	int returnCode = system(commandLine.c_str());

	return 0;
}
