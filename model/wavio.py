import wave
import struct

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

# Example usage:
if __name__ == "__main__":
	bin_to_wav("../data/iq_samples.bin", "../data/audio_processed.wav", sample_width=2, num_channels=1, sample_rate=48e3)
