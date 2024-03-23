/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

typedef struct {
    unsigned short mono_upsample;
    unsigned short mono_downsample;
} TsMonoConfig;

typedef struct {
    unsigned short stereo_upsample;
    unsigned short downsample;
    // add more as required for stereo
} TsStereoConfig;

typedef struct {
    unsigned int   block_size;
    uint8_t        rf_downsample;
    TsMonoConfig   mono;
    TsStereoConfig stereo;
} TsConfig;

#endif // CONFIG_H