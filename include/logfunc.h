/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef DY4_LOGFUNC_H
#define DY4_LOGFUNC_H

// add headers as needed
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

void genIndexVector(std::vector<float> &, \
	const int);

void logVector(const std::string, \
	const std::vector<float> &, \
	const std::vector<float> &);

template<typename T>
void logVector(const std::string filename, \
	const std::vector<T> &y)
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



#endif // DY4_LOGFUNC_H
