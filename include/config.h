/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

struct AudioConfig {
    unsigned short upsample;
    unsigned short downsample;
};

struct RdsConfig {
    unsigned short upsample;
    unsigned short downsample;
    unsigned short sps;
};

struct Config {
    unsigned int   block_size;
    uint8_t        rf_downsample;
    AudioConfig    audio;
    RdsConfig      rds;
};

void argparse_mode_channel(int argc, char* argv[], int& mode, int& channel)
{
    if (argc < 2) {
		std::cerr << "Operating in default mode 0 and channel 0 (mono)" << std::endl;
	} else if (argc == 2 || argc == 3) {
		mode = std::atoi(argv[1]);
		if (mode > 3 || mode < 0) {
			std::cerr << "Invalid mode entered: " << mode << std::endl;
			exit(1);
		}
	}

	if (argc == 3) {
		if (std::string(argv[2]) == "m") {
			channel = 0;
		} else if (std::string(argv[2]) == "s") {
			channel = 1;
		} else if (std::string(argv[2]) == "r") {
			channel = 2;
			if (mode == 1 || mode == 3) {
				std::cerr << "Invalid channel for rds entered: " << argv[2] << std::endl;
				exit(1);
			}
		}
		else {
		std::cerr << "Invalid channel entered: " << argv[2] << std::endl;
		exit(1);
		}
	} else if (argc > 3) {
		std::cerr << "Usage: " << argv[0] << std::endl;
		std::cerr << "or " << argv[0] << " <mode>" << std::endl;
		std::cerr << "\t\t <mode> is a value from 0 to 3" << std::endl;
		exit(1);
	}


	if (channel == 0) {
		std::cerr << "Operating in mode " << mode << " with mono channel" << std::endl;
	} else if (channel == 1) {
		std::cerr << "Operating in mode " << mode << " with stereo channel" << std::endl;
	} else if (channel == 2) {
		std::cerr << "Operating in mode " << mode << " with rds channel" << std::endl;
	}
}

#endif // CONFIG_H