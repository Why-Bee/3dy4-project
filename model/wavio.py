import numpy as np
from scipy.io import wavfile

if __name__ == "__main__":
	audio_Fs = 48e3;
    # input binary file name (from where samples are read into Python)
	# the default is JUST a SELF-CHECK; of course, change filenames as needed
	in_fname = "../data/iq_samples.bin"
	# in_fname = "../data/float32filtered.bin"
	# read data from a binary file (assuming 32-bit floats)
	float_data = np.fromfile(in_fname, dtype='int16')
	print(" Read binary data from \"" + in_fname + "\" in int16 format")

	# we assume below there are two audio channels where data is
	# interleaved, i.e., left channel sample, right channel sample, ...
	# for mono .wav files the reshaping below is unnecessary
	# reshaped_data = np.reshape(float_data, (-1, 2))

	

	wavfile.write("../data/audio_processed.wav", \
				audio_Fs, \
				float_data.astype(np.int16))

	# note: we can also dump audio data in other formats, if needed
	# audio_data.astype('int16').tofile('int16samples.bin')
