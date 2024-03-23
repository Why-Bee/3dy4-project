#include <gtest/gtest.h>
#include <cstdlib> // For system function
#include <string>
#include <fstream>
#include <iostream>
#include "filter.h"
#include "pll.h"

#define EPSILON_GROUP30 1e-1

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
TEST_F(DatFileComparisonTestStereo, DISABLED_DemodSamplesSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_demodulated_samples1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/demodulated_samples1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

// Test case to compare .dat files
TEST_F(DatFileComparisonTestStereo, DISABLED_DemodSamplesDelayedSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_demodulated_samples_delayed1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/demodulated_samples_delayed1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_FloatMonoDataSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_mono_data1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_mono_data1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_PilotFilteredSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_pilot_filtered1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/pilot_filtered1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_NcoOutSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_nco_out1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/nco_out1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_StereoMixedSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_stereo_mixed1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/stereo_mixed1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_StereoLpfSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_stereo_lpf_filtered1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/stereo_lpf_filtered1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_FloatStereoLeftDataSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_stereo_left_data1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_stereo_left_data1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_FloatStereoRightDataSameAsModelBlock1) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_stereo_right_data1.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_stereo_right_data1.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 1; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

// Test case to compare .dat files
TEST_F(DatFileComparisonTestStereo, DISABLED_DemodSamplesSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_demodulated_samples10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/demodulated_samples10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

// Test case to compare .dat files
TEST_F(DatFileComparisonTestStereo, DISABLED_DemodSamplesDelayedSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_demodulated_samples_delayed10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/demodulated_samples_delayed10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_FloatMonoDataSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_mono_data10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_mono_data10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_PilotFilteredSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_pilot_filtered10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/pilot_filtered10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_NcoOutSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_nco_out10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/nco_out10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_StereoMixedSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_stereo_mixed10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/stereo_mixed10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_StereoLpfSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_stereo_lpf_filtered10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/stereo_lpf_filtered10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_FloatStereoLeftDataSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_stereo_left_data10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_stereo_left_data10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, DISABLED_FloatStereoRightDataSameAsModelBlock10) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_float_stereo_right_data10.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/float_stereo_right_data10.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());
    ASSERT_GT(expected_data.size(), 0);

    // Iterate over each pair of values and perform ASSERT_NEAR
    for (size_t i = 10; i < expected_data.size(); ++i) {
        ASSERT_NEAR(expected_data[i], actual_data[i], EPSILON_GROUP30); // Adjust EPSILON_GROUP30 as needed
    }
}

TEST_F(DatFileComparisonTestStereo, PllPyCompareBlock100) {
    constexpr float kPilotToneFrequency = 19e3;
    constexpr float kPilotNcoScale = 2.0;
    constexpr float kFs = 240000.0;
    int mismatch_count = 0;

    std::vector<float> pllStatesIn = readDatFile<float>("../data/py_pll_state_in100.dat");

    PllState pllState;

    pllState.integrator = pllStatesIn[0];
    pllState.phaseEst = pllStatesIn[1];
    pllState.feedbackI = pllStatesIn[2];
    pllState.feedbackQ = pllStatesIn[3];
    pllState.trigOffset = pllStatesIn[4];
    pllState.lastNco = pllStatesIn[5];

    std::vector<float> pilot_filtered = readDatFile<float>("../data/py_pilot_filtered100.dat");

    std::vector<float> actual_nco_out(pilot_filtered.size(), 0.0);

    fmPll(pilot_filtered,
           kPilotToneFrequency, 
           kFs,
           pllState,
           kPilotNcoScale,
           actual_nco_out);

    
    std::vector<float> expected_nco_out = readDatFile<float>("../data/py_nco_out200.dat");

    std::vector<float> pllStatesOutExpected = readDatFile<float>("../data/py_pll_state_out200.dat");

    std::vector<float> pllStatesOutActual(6, 0.0);

    PllState expected_pll_state;

    pllStatesOutActual[0] = pllState.integrator;
    pllStatesOutActual[1] = pllState.phaseEst;
    pllStatesOutActual[2] = pllState.feedbackI;
    pllStatesOutActual[3] = pllState.feedbackQ;
    pllStatesOutActual[4] = pllState.trigOffset;
    pllStatesOutActual[5] = pllState.lastNco;

    // for (size_t i = 0; i<6; i++)
    //     EXPECT_NEAR(pllStatesOutExpected[i], pllStatesOutActual[i], 1e-1);

    for (size_t i = 0; i<actual_nco_out.size(); i++) {
        EXPECT_NEAR(actual_nco_out[i], expected_nco_out[i], 1.5e-2) <<
        "i: " << i << " mismatch_count: " << ++mismatch_count;
    }
}


TEST_F(DatFileComparisonTestStereo, DISABLED_CheckDelay) {
    size_t delay_size = 20;

    std::vector<float> idx_100 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};

    std::vector<float> idx_100_exp = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79 };

    std::vector<float> idx_delayed;
    std::vector<float> idx_delayed_state(delay_size, 0.0);

    delayBlock(idx_100, idx_delayed, idx_delayed_state);

    for (int i = 0; i < idx_100_exp.size(); i++) {
        ASSERT_NEAR(idx_100_exp[i], idx_delayed[i], 1e-5);
    }
}




