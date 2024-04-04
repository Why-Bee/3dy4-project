/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "logfunc.h"

// function to generate a vector whose value is equal to its index
// this is useful when plotting a vector because we use the index on the X axis
void genIndexVector(std::vector<float> &x, const int size) {
	x.clear(); x.resize(size, float(0));
	for (int i=0; i<size; i++) {
		x[i] = float(i);
	}
}

// function to be used for logging a float vector in a .dat file (for .gnuplot)
// can be reused for different types of vectors with 32-bit floating point vals
void logVector(const std::string filename, \
	const std::vector<float> &x, \
	const std::vector<float> &y)
{
	// write data in text format to be parsed by gnuplot (change as needed)
	const std::string dat_filename = "../data/" + filename + ".dat";
	std::fstream fd;
	fd.open(dat_filename, std::ios::out);
	fd << "#\tx_axis\ty_axis\n";

	for (int i = 0; i < (int)x.size(); i++) {
		fd << "\t " << x[i] << "\t";
		// if the number of values on the Y axis is less than on the X tx_axis
		// then we just do not write anything on the Y axis
		if (i < (int)y.size())
			fd << y[i];
		fd << "\n";
	}
	std::cout << "Generated " << dat_filename << " to be used by gnuplot\n";
	fd.close();
}

// function to be used for logging a float vector in a .dat file (for .gnuplot)
// can be reused for different types of vectors with 32-bit floating point vals
void logVector(const std::string filename, \
	const std::vector<float> &y)
{
	std::vector<float> x;
	genIndexVector(x, y.size());
	// write data in text format to be parsed by gnuplot (change as needed)
	const std::string dat_filename = "../data/" + filename + ".dat";
	std::fstream fd;
	fd.open(dat_filename, std::ios::out);
	fd << "#\tx_axis\ty_axis\n";

	for (int i = 0; i < (int)x.size(); i++) {
		fd << "\t " << x[i] << "\t";
		// if the number of values on the Y axis is less than on the X tx_axis
		// then we just do not write anything on the Y axis
		if (i < (int)y.size())
			fd << y[i];
		fd << "\n";
	}
	std::cout << "Generated " << dat_filename << " to be used by gnuplot\n";
	fd.close();
}

void logVectorTiming(const std::string filename, \
	std::vector<float> &y,
	int curr_block,
	int start_block,
	int num_blocks,
	float data) {
	if (curr_block >= start_block && curr_block < start_block+num_blocks) {
		y[curr_block-start_block] = data;
	} else if (curr_block ==  start_block+num_blocks) { // Dump data
		logVector("timing/" + filename, y);
	}
}
