#include <gtest/gtest.h>
#include <cstdlib> // For system function
#include <string>
#include <fstream>
#include <iostream>
#include "filter.h"

#define EPSILON_GROUP30 1e-2

// Define the test fixture class
class DatFileComparisonTestStereo : public ::testing::Test {
public:
    DatFileComparisonTestStereo() {}
protected:
    // Function to run Python script and generate actual .dat file
    void runModel(std::string model_command) {
        std::system(model_command.c_str());
    }

    template<typename T>
    std::vector<T> readDatFile(const std::string& file_path) {
        std::vector<T> values;
        std::ifstream file(file_path);
        std::string line;
        std::getline(file, line); // Skip header line
        T x, y;
        while (file >> x >> y) {
            values.push_back(y);
        }
        return values;
    }
};

// Test case to compare .dat files
TEST_F(DatFileComparisonTestStereo, DemodSamplesSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_demodulated_samples10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/demodulated_samples10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

// Test case to compare .dat files
TEST_F(DatFileComparisonTestStereo, DemodSamplesDelayedSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_demodulated_samples_delayed10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/demodulated_samples_delayed10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, FloatMonoDataSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_mono_data10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_mono_data10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, PilotFilteredSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_pilot_filtered10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/pilot_filtered10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, NcoOutSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_nco_out10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/nco_out10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, StereoMixedSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_stereo_mixed10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/stereo_mixed10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, StereoLpfSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_stereo_lpf_filtered10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/stereo_lpf_filtered10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, FloatStereoLeftDataSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_stereo_left_data10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_stereo_left_data10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, FloatStereoRightDataSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_stereo_right_data10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_stereo_right_data10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}


