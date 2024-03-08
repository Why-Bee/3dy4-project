import wave
import struct
import sys
from scipy.io import wavfile
import numpy as np

def v2_write_wav(in_fname):
    audio_Fs = 48e3
    # input binary file name (from where samples are read into Python)
    # the default is JUST a SELF-CHECK; of course, change filenames as needed
    # in_fname = "../data/float32filtered.bin"
    # read data from a binary file (assuming 32-bit floats)
    data = np.fromfile(in_fname, dtype='int16')
    print(" Read binary data from \"" + in_fname + "\" in int16 format")


    # self-check if the read and write are working correctly
    # not needed while working with data generated from C++

    wavfile.write("../data/audio_processed.wav", \
                audio_Fs, \
                data.astype(np.int16))
    

def bin_to_wav(input_file, output_file, sample_width=2, num_channels=1, sample_rate=44100):
    with open(input_file, 'rb') as f:
        # Read binary data from the input file
        bin_data = f.read()

    # Determine number of frames
    num_frames = len(bin_data) // sample_width // num_channels

    # Open the output WAV file
    with wave.open(output_file, 'w') as wav_file:
        # Set WAV file parameters
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        
        # Convert binary data to WAV format and write to WAV file
        wav_data = struct.unpack('<' + 'h' * (num_frames * num_channels), bin_data)
        wav_file.writeframesraw(struct.pack('<' + 'h' * (num_frames * num_channels), *wav_data))

def v2bin_to_wav(input_file, output_file, sample_width=2, num_channels=1, sample_rate=44100):
    with open(input_file, 'rb') as f:
        # Read binary data from the input file
        bin_data = f.read()

    # Determine number of frames
    num_frames = len(bin_data) // sample_width // num_channels

    # Convert binary data to numpy array of int16
    wav_data = np.frombuffer(bin_data, dtype=np.int16)

    # Open the output WAV file and write the data
    with wave.open(output_file, 'w') as wav_file:
        # Set WAV file parameters
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)

        # Write the data to the WAV file
        wav_file.writeframes(wav_data.tobytes())

# Example usage:
if __name__ == "__main__":
    input_file = "../data/iq_samples.bin"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    output_file = "../data/audio_processed.wav"
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    #v2_write_wav(input_file)
    bin_to_wav(input_file, output_file, sample_width=2, num_channels=1, sample_rate=48e3)
    # v2bin_to_wav(input_file, output_file, sample_width=2, num_channels=1, sample_rate=48e3)
