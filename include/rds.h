#include <vector> 
#include <unordered_map>
#pragma once

constexpr uint8_t kCheckLen = 26;
constexpr uint8_t kBadSyndromeScore = 10;
constexpr uint8_t kGoodSyndromeScore = 10;

 // Functions

int sampling_start_adjust(const std::vector<float> &block, 
                           const int samples_per_symbol);

void symbol_vals_to_bits(std::vector<bool>& bool_array, 
                         int& ll_count,
                         int& hh_count, 
                         const std::vector<float>& sampling_points, 
                         const int offset, 
                         const float last_value_state);

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
                        int& state_len);

void frame_sync_blockwise(const std::vector<bool>& bitstream,
                          char& expected_next,
                          uint16_t& rubish_score,
                          uint16_t& rubbish_streak,
                          std::vector<bool>& state_values,
                          int& state_len,
                          uint32_t& ps_next_up,
                          uint32_t& ps_next_up_pos,
                          uint8_t& ps_num_chars_set,
                          std::string& program_service);