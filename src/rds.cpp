#include <vector>
#include "rds.h"

int sampling_start_adjust(const std::vector<float> &block, const int samples_per_symbol) {
    int abs_min_idx = 0;
    float abs_min = std::abs(block[abs_min_idx]);
    float diff;
    for (int i = 0; i < block.size()-10; i++) {
        diff = std::abs(block[i]);
        if (diff < abs_min) {
            abs_min = diff;
            abs_min_idx = i;
        }
    }

    return ((abs_min_idx + static_cast<int>(samples_per_symbol/2)) % samples_per_symbol);
}


// modifies: bool-array, ll_count, hh_count
void symbol_vals_to_bits(std::vector<bool>& bool_array, 
                         int& ll_count,
                         int& hh_count, 
                         const std::vector<float>& sampling_points, 
                         const int offset, 
                         const int last_value_state) 
{
    bool_array.resize(sampling_points.size()/2);
    hh_count = 0;
    ll_count = 0;
    bool first_val = 0;
    bool second_val = 0;
    for (int i = 0; i < sampling_points.size()-1; i+=2) {
        if ((i+offset)-1 < 0) {
            first_val = last_value_state;
        } else {
            first_val = sampling_points[(i+offset)-1];
        }
        second_val = sampling_points[(i+offset)];

        if (first_val == 0 && second_val == 0) {
            std::cerr << "DOUBLE ZERO WARNING" << std::endl;
            bool_array[i/2] = false;
        }

        if (first_val >= 0 && second_val <= 0) { // case HL
            bool_array[i/2] = true;
        } else if (first_val <= 0 && second_val >= 0) { // case LH
            bool_array[i/2] = false;
        } else if (first_val < 0 && second_val < 0) { // case LL check weaks
            if (first_val < second_val) { // weak LH
                bool_array[i/2] = false;
            } else if (first_val >= second_val) { // weak HL
                bool_array[i/2] = true;
            }
            ll_count++;
        } else if (first_val > 0 && second_val > 0) { // case HH
            if (first_val > second_val) { // weak HL
                bool_array[i/2] = true;
            } else if (first_val <= second_val) { // weak LH
                bool_array[i/2] = false;
            }
            hh_count++;
        }
    }
}

// modifies: decoded, last_val_state
void differential_decode_stateful(std::vector<bool>& decoded, 
                                  bool& last_val_state, 
                                  const std::vector<bool>& bool_array) 
{
    decoded.resize(bool_array.size()-1);
    decoded[0] = last_val_state ^ bool_array[0];
    for (int i = 1; i < bool_array.size()-1; i++) {
        decoded[i] = bool_array[i] ^ bool_array[i-1];
    }
    last_val_state = bool_array[bool_array.size()-1];
}

uint32_t concat_bool_arr(const std::vector<bool>& bool_arr) {
    uint32_t result = 0;
    for (bool bit : bool_arr) {
        result = (result << 1) | static_cast<uint32_t>(bit);
    }
    return result;
}

uint32_t multiply_parity(const std::vector<bool>& matrix1) {
    std::vector<bool> result(parity_matrix[0].size(), false);
    for (int j = 0; j < parity_matrix[0].size(); j++) {
        for (int k = 0; k < parity_matrix.size(); k++) {
            result[j] ^= matrix1[k] & parity_matrix[k][j];
        }
    }
    return concat_bool_arr(result);
}

std::pair<bool, char> matches_syndrome(uint32_t ten_bit_val) {
    std::map<uint32_t. char> les_syndromes = {
        {0b1111011000, 'A'},
        {0b1111010100, 'B'},
        {0b1001011100, 'C'},
        {0b1111001100, 'P'}, // Cprime
        {0b1001011000, 'D'}
    };

    for (auto& [value, syndrome] : les_syndromes) {
        if (ten_bit_val == value) {
            return {true, syndrome};
        }
    }

    return {false, ' '};
}

// modifies: bitstream, bitstream_select, bitstream_score_0, bitstream_score_1, last_value_state
// TODO: consider default
void recover_bitstream(std::vector<bool>& bitstream, 
                          int& bitstream_select, 
                          int& bitstream_score_0, 
                          int& bitstream_score_1, 
                          bool& last_value_state, 
                          const std::vector<float>& sampling_points, 
                          const int bitstream_select_thresh) 
    {
     int ll_count0, hh_count0, ll_count1, hh_count1;
     if (bitstream_select == 0) {
          symbol_vals_to_bits(bitstream, ll_count0, hh_count0, sampling_points, 0, last_value_state);
     } else if (bitstream_select == 1) {
          symbol_vals_to_bits(bitstream, ll_count1, hh_count1, sampling_points, 1, last_value_state);
     } else if (bitstream_select == -1) {
          std::vector<bool> bitstream0;
          std::vector<bool> bitstream1;
          symbol_vals_to_bits(bitstream0, ll_count0, hh_count0, sampling_points, 0, last_value_state);
          symbol_vals_to_bits(bitstream1, ll_count1, hh_count1, sampling_points, 1, last_value_state);
          if ((ll_count0+hh_count0) < (ll_count1+hh_count1)) {
                bitstream_score_0 += 1;
                bitstream_score_1 -= 1;
                if (bitstream_score_0 >= bitstream_select_thresh) {
                    std::cerr << "SELECTING BITSTREAM 0" << std::endl;
                    bitstream_select = 0;
                    bitstream = bitstream0;
                }
          } else {
                bitstream = bitstream1;
                bitstream_score_1 += 1;
                bitstream_score_0 -= 1;
                if (bitstream_score_1 >= bitstream_select_thresh) {
                    std::cerr << "SELECTING BITSTREAM 1" << std::endl;
                    bitstream_select = 1;
                }
          }
     }
     last_value_state = bitstream[bitstream.size()-1];
    }
