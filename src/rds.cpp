#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>

#include "rds.h"


// Values
static const std::vector<std::vector<bool>> parity_matrix = 
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


static std::unordered_map<char, char> next_syndrome_dict{
                                              {'A', 'B'}, 
                                              {'B', 'C'}, 
                                              {'C', 'D'}, 
                                              {'D', 'A'}};

const std::unordered_map<uint32_t, std::string> program_type_dict{
                                              {0, "None"}, 
                                              {1, "News"}, 
                                              {2, "Information"}, 
                                              {3, "Sports"},
                                              {4, "Talk"},
                                              {5, "Rock"},
                                              {6, "Classic Rock"},
                                              {7, "Adult Hits"},
                                              {8, "Soft Rock"},
                                              {9, "Top 40"},
                                              {10, "Country"}};

int sampling_start_adjust(const std::vector<float> &block, const int samples_per_symbol) {
    int abs_min_idx = 0;
    float abs_min = std::abs(block[abs_min_idx]);
    float diff;
    for (int i = 0; i < block.size(); i++) {
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
                         const float last_value_state) 
{
    bool_array.resize(sampling_points.size()/2);
    hh_count = 0;
    ll_count = 0;
    float first_val = 0.0;
    float second_val = 0.0;
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
    decoded.resize(bool_array.size());
    decoded[0] = last_val_state ^ bool_array[0];
    for (int i = 1; i < bool_array.size(); i++) {
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
            result[j] = result[j] ^ (matrix1[k] && parity_matrix[k][j]);
        }
    }
    return concat_bool_arr(result);
}

std::pair<bool, char> matches_syndrome(uint32_t ten_bit_val) {
    std::unordered_map<uint32_t, char> les_syndromes = {
        {0b1111011000, 'A'},
        {0b1111010100, 'B'},
        {0b1001011100, 'C'},
        {0b1111001100, 'C'}, // Cprime
        {0b1001011000, 'D'}
    };
 
    auto syndrome = les_syndromes.find(ten_bit_val);
    if (syndrome != les_syndromes.end()) {
        return {true, syndrome->second};
    } else {
        return {false, ' '};
    }
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
    int ll_count0, hh_count0, ll_count1, hh_count1 = 0;
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
        } 
        else {
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


/* cpp implementation of the following code:
def frame_sync_initial(bitstream, found_count, last_found_counter, expected_next, state_values, state_len):
    CHECK_LEN = 26
 
    for start_idx in range(-state_len, len(bitstream)-CHECK_LEN):
        if start_idx < 0:
            twenty_six_bit_value = np.concatenate((state_values[state_len+start_idx:], bitstream[:CHECK_LEN+start_idx]))
        else:
            twenty_six_bit_value = bitstream[start_idx:start_idx+CHECK_LEN]
            
        ten_bit_code = multiply_parity(twenty_six_bit_value)
  
        is_valid, syndrome = matches_syndrome(ten_bit_code)
        if is_valid:
            print("Found Syndrome", syndrome)
            if found_count > 0:
                if syndrome != expected_next and last_found_counter == CHECK_LEN:
                    print(f"FALSE SYNDROME: not expected next ({syndrome}), ", last_found_counter)
                    found_count = 0
                    expected_next = None
                    continue
                elif syndrome != expected_next:
                    continue
                
            expected_next = next_syndrome_dict[syndrome]
            found_count += 1
            last_found_counter = 0
            print("Good Syndrome,", found_count)

        last_found_counter+=1
        if last_found_counter > CHECK_LEN:
            print(f"Did not find the next syndrome {last_found_counter}, resetting")
            last_found_counter = 0
            found_count = 0
            expected_next = None
  
    state_len = CHECK_LEN-1
    next_state = bitstream[-state_len:]
 
    return found_count, last_found_counter, expected_next, next_state, state_len
*/

void frame_sync_initial(const std::vector<bool>& bitstream, 
                        int& found_count, 
                        int& last_found_counter, 
                        char& expected_next, 
                        std::vector<bool>& state_values, 
                        int& state_len)
{
    std::vector<bool> twenty_six_bit_value(26, 0);
    uint32_t ten_bit_code;
    char syndrome = '_';
    bool is_valid = false;

    for(int start_idx = -state_len; start_idx < static_cast<int>(bitstream.size()-kCheckLen); start_idx++) {
        if (start_idx < 0) {
            int j = 0;
            for (int i = state_len+start_idx; i < state_len; i++, j++) {
                twenty_six_bit_value[j] = state_values[i];
            }
            //std::cerr << start_idx + kCheckLen << std:: endl;
            for (int i = 0; i < start_idx+kCheckLen; i++, j++) {
                twenty_six_bit_value[j] = bitstream[i<0?(bitstream.size()-i) : i];
            }
        } else {
            int j = 0;
            for (int i = start_idx; i < kCheckLen+start_idx; i++, j++)
                twenty_six_bit_value[j] = bitstream[i];
        }

        ten_bit_code = multiply_parity(twenty_six_bit_value);

        std::tie(is_valid, syndrome) = matches_syndrome(ten_bit_code);
        if (is_valid) {
            //std::cerr<<"Found Syndrome: " << syndrome  << std::endl;
            if (found_count > 0) {
                if (syndrome != expected_next && last_found_counter == kCheckLen) {
                    //std::cerr<< "False Syndrome: not expected next (" << syndrome << ") " << last_found_counter << std::endl;
                    found_count = 0;
                    expected_next = '\0';
                    continue;
                }
                else if (syndrome != expected_next){
                    continue;
                }
            }
            expected_next = next_syndrome_dict[syndrome];
            found_count++;
            last_found_counter = 0;
            //std::cerr << "Good syndrome, " << found_count << std::endl;
        }
        last_found_counter += 1;
        if (last_found_counter > kCheckLen) {
            //std::cerr << "did not find next syndrome " << last_found_counter << " ,resetting" << std::endl;
            last_found_counter = 0;
            found_count = 0;
            expected_next = '\0';
        }
    }

    state_len = kCheckLen-1;

    for (int i = bitstream.size()-state_len, j = 0; i < bitstream.size(); i++, j++) {
        state_values[j] = bitstream[i];
    }
}

void frame_sync_blockwise(const std::vector<bool>& bitstream,
                          char& expected_next,
                          uint16_t& rubbish_score,
                          uint16_t& rubbish_streak,
                          std::vector<bool>& state_values,
                          int& state_len,
                          uint32_t& ps_next_up,
                          uint32_t& ps_next_up_pos,
                          uint8_t& ps_num_chars_set,
                          std::string& program_service) {

    static int pi = -1;
    static std::string pty = "None", ps = "None";

    uint16_t ten_bit_code;

    std::vector<bool> twenty_six_bit_value;
    twenty_six_bit_value.reserve(kCheckLen);

    for(int start_idx = -state_len; start_idx < static_cast<int>(bitstream.size()-kCheckLen); start_idx+=kCheckLen) {
        if (start_idx < 0) {
            int j = 0;
            for (int i = state_len+start_idx; i < state_len; i++, j++) {
                twenty_six_bit_value[j] = state_values[i];
            }
            //std::cerr << start_idx + kCheckLen << std:: endl;
            for (int i = 0; i < start_idx+kCheckLen; i++, j++) {
                twenty_six_bit_value[j] = bitstream[i<0?(bitstream.size()-i) : i];
            }
        } else {
            int j = 0;
            for (int i = start_idx; i < kCheckLen+start_idx; i++, j++)
                twenty_six_bit_value[j] = bitstream[i];
        }

        ten_bit_code = multiply_parity(twenty_six_bit_value);

       auto[is_valid, syndrome] = matches_syndrome(ten_bit_code);

       if (!is_valid || syndrome != expected_next) {
            //std::cerr << "bad syndrome " << ++bad_syndrome_count << std::endl;
            // order is important here
            rubbish_streak++;
            rubbish_score += kBadSyndromeScore*rubbish_streak;
            syndrome = expected_next;
       } else { 
            //std::cerr << "good syndrome " << syndrome << std::endl;

            rubbish_streak = 0;
            if (rubbish_score > 0) {
                rubbish_score -= kGoodSyndromeScore;
            }
       }
        if (rubbish_streak == 0) {
            std::cerr << "PROGRAM INFO:" << "\r";
            
            if (syndrome == 'A') {   
                pi = concat_bool_arr(std::vector<bool>(twenty_six_bit_value.begin(), twenty_six_bit_value.begin() + 16));                               
            }
            if (syndrome == 'B') {
                pty = program_type_dict.find(concat_bool_arr((std::vector<bool>(twenty_six_bit_value.begin() + 6,
                                                                        twenty_six_bit_value.begin() + 11))))->second;                
                ps_next_up = concat_bool_arr(std::vector<bool>(twenty_six_bit_value.begin(), 
                                                            twenty_six_bit_value.begin() + 5));
                ps_next_up_pos = concat_bool_arr(std::vector<bool>(twenty_six_bit_value.begin() + 14, 
                                                                twenty_six_bit_value.begin() + 16));
        }
            if (syndrome == 'D') {
                if (ps_next_up == 0 || ps_next_up == 1) {
                    if (ps_num_chars_set < 8) {
                        ps_num_chars_set += 2;
                    }

                    program_service = program_service.substr(0, 2*ps_next_up_pos) +
                                    static_cast<char>(concat_bool_arr(std::vector<bool>(twenty_six_bit_value.begin(), twenty_six_bit_value.begin() + 8))) +
                                    static_cast<char>(concat_bool_arr(std::vector<bool>(twenty_six_bit_value.begin() + 8, twenty_six_bit_value.begin() + 16))) +
                                    program_service.substr(2*ps_next_up_pos + 2, program_service.length()-1);

                    if (ps_num_chars_set == 8 && ps_next_up_pos == 0b11) {
                        ps_num_chars_set = 0;
                        ps = program_service;
                    }
                }
            }

            std::cerr << "PROGRAM INFORMATION" << std::endl;
            std::cerr << "PI: " << std::hex << pi << std::endl;
            std::cerr << "PTY: " << pty << std::endl;
            std::cerr << "PROGRAM SERVICE: " << ps << std::endl << std::endl;
        }
       expected_next = next_syndrome_dict.at(syndrome);
    }
    state_len = (bitstream.size() + state_len) % kCheckLen;
    state_values = std::vector<bool>(bitstream.end() - state_len, bitstream.end());
}
