#include "mono.h"
#include <vector>

class MonoBlockProcessor {
    private:

    public:
        void ProcessBlock() {
            convolveFIR2(float_audio_data, 
                        demodulated_samples,
                        mono_coeffs, 
                        mono_state,
                        kMonoDecimation);

            s16_audio_data.clear();
            for (unsigned int k = 0; k < float_audio_data.size(); k++){
                    if (std::isnan(float_audio_data[k])) s16_audio_data.push_back(0);
                    else s16_audio_data.push_back(static_cast<short int>(float_audio_data[k]*(kMaxUint14+1)));
            }
            
            fwrite(&s16_audio_data[0], sizeof(short int), s16_audio_data.size(), stdout);
        }
};
