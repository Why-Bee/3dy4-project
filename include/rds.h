#include <vector> 
#include <unordered_map>
#pragma once


// Values
const std::vector<std::vector<bool>> parity_matrix = 
{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 
 {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
 {1, 0, 1, 1, 0, 1, 1, 1, 0, 0},
 {0, 1, 0, 1, 1, 0, 1, 1, 1, 0},
 {0, 0, 1, 0, 1, 1, 0, 1, 1, 1},
 {1, 0, 1, 0, 0, 0, 0, 1, 1, 1},
 {1, 1, 1, 0, 0, 1, 1, 1, 1, 1},
 {1, 1, 0, 0, 0, 1, 0, 0, 1, 1},
 {1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
 {1, 1, 0, 1, 1, 1, 0, 1, 1, 0},
 {0, 1, 1, 0, 1, 1, 1, 0, 1, 1},
 {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
 {1, 1, 1, 1, 0, 1, 1, 1, 0, 0},
 {0, 1, 1, 1, 1, 0, 1, 1, 1, 0},
 {0, 0, 1, 1, 1, 1, 0, 1, 1, 1},
 {1, 0, 1, 0, 1, 0, 0, 1, 1, 1},
 {1, 1, 1, 0, 0, 0, 1, 1, 1, 1},
 {1, 1, 0, 0, 0, 1, 1, 0, 1, 1}};

constexpr uint8_t kCheckLen = 26;
constexpr uint8_t kBadSyndromeScore = 10;
constexpr uint8_t kGoodSyndromeScore = 10;

const std::unordered_map<char, char> next_syndrome_dict{{'A', 'B'}, 
                                              {'B', 'C'}, 
                                              {'C', 'D'}, 
                                              {'P', 'D'}, 
                                              {'D', 'A'}};
 // Functions

int sampling_start_adjust(const std::vector<float> &block, 
                           const int samples_per_symbol);

void symbol_vals_to_bits(std::vector<bool>& bool_array, 
                         int& ll_count,
                         int& hh_count, 
                         const std::vector<float>& sampling_points, 
                         const int offset, 
                         const int last_value_state);

void differential_decode_stateful(std::vector<bool>& decoded, 
                                  bool& last_val_state, 
                                  const std::vector<bool>& bool_array);

uint32_t concat_bool_arr(const std::vector<bool>& bool_arr);

uint32_t multiply_parity(const std::vector<bool>& matrix1);

std::pair<bool, char> matches_syndrome(uint32_t ten_bit_val);

void recover_bitstream(std::vector<bool>& bitstream, 
                          int& bitstream_select, 
                          int& bitstream_score_0, 
                          int& bitstream_score_1, 
                          bool& last_value_state, 
                          const std::vector<float>& sampling_points, 
                          const int bitstream_select_thresh);

void frame_sync_initial(std::vector<bool> bitstream, 
                        int& found_count, 
                        int& last_found_counter, 
                        char& expected_next, 
                        std::vector<bool> state_values, 
                        int& state_len,
                        std::vector<bool> next_state);