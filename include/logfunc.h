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
#include <chrono>

void genIndexVector(std::vector<float> &, \
	const int);

void logVector(const std::string, \
	const std::vector<float> &, \
	const std::vector<float> &);

void logVector(const std::string filename, \
	const std::vector<float> &y);

void logVectorTiming(
	const std::string filename,
	std::vector<float> &y,
	int curr_block,
	int start_block,
	int num_blocks,
	float data);

template<typename Func>
float timeFunction(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func(); // Execute the function
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration in milliseconds
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count();
}

#endif // DY4_LOGFUNC_H
