#include <gtest/gtest.h>
#include <cstdlib> // For system function
#include <string>
#include <fstream>
#include <iostream>

// Define the test fixture class
class DatFileComparisonTest : public ::testing::Test {
public:
    DatFileComparisonTest() {}
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
TEST_F(DatFileComparisonTest, RfImpulseSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_impulse_resp_rf.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/impulse_resp_rf.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], 1e-3); // Adjust epsilon as needed
    }
}

// Test case to compare .dat files
TEST_F(DatFileComparisonTest, MonoImpulseSameAsModel) {
    // Run Python script to generate actual .dat file
    std::vector<float> expected_data = readDatFile<float>("../data/py_impulse_resp_mono.dat");
    std::vector<float> actual_data = readDatFile<float>("../data/impulse_resp_mono.dat");

    // Assert that the sizes of both vectors are equal
    ASSERT_EQ(expected_data.size(), actual_data.size());

    // Iterate over each pair of values and perform EXPECT_NEAR
    for (size_t i = 0; i < expected_data.size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], 1e-5); // Adjust epsilon as needed
    }
}

// Test case to compare .dat files
TEST_F(DatFileComparisonTest, IPreFmDemodFirst3Blocks) {
    // Run Python script to generate actual .dat file
    for (int i = 0; i < 3; i++) {
        std::vector<float> expected_data = readDatFile<float>(
            "../data/py_pre_fm_demod_i"+std::to_string(i)+".dat");
        std::vector<float> actual_data = readDatFile<float>(
            "../data/pre_fm_demod_i"+std::to_string(i)+".dat");

        // Assert that the sizes of both vectors are equal
        ASSERT_EQ(expected_data.size(), actual_data.size());

        // Iterate over each pair of values and perform EXPECT_NEAR
        for (size_t i = 0; i < expected_data.size(); ++i) {
            EXPECT_NEAR(expected_data[i], actual_data[i], 1e-5); // Adjust epsilon as needed
        }
    }
}

// Test case to compare .dat files
TEST_F(DatFileComparisonTest, QPreFmDemodFirst3Blocks) {
    // Run Python script to generate actual .dat file
    for (int i = 0; i < 3; i++) {
        std::vector<float> expected_data = readDatFile<float>(
            "../data/py_pre_fm_demod_q"+std::to_string(i)+".dat");
        std::vector<float> actual_data = readDatFile<float>(
            "../data/pre_fm_demod_q"+std::to_string(i)+".dat");

        // Assert that the sizes of both vectors are equal
        ASSERT_EQ(expected_data.size(), actual_data.size());

        // Iterate over each pair of values and perform EXPECT_NEAR
        for (size_t i = 0; i < expected_data.size(); ++i) {
            ASSERT_NEAR(expected_data[i], actual_data[i], 1e-3); // Adjust epsilon as needed
        }
    }
}