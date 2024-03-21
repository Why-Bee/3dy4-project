#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Copyright by Nicola Nicolici
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import math

# use fmDemodArctan and fmPlotPSD
from fmSupportLib import fmDemodArctan, fmPlotPSD, own_lfilter, lpCoeff, custom_fm_demod, logVector, bpFirwin, fmPll, delayBlock
# for take-home add your functions

rf_Fs = 2.4e6
rf_Fc = 100e3
rf_taps = 101
rf_decim = 10

stereo_bp_fc_low = 22e3
stereo_bp_fc_high = 54e3
stereo_bp_taps = 101

stereo_bp_fc_low = 22e3
stereo_bp_fc_high = 54e3
stereo_bp_taps = 101

pilot_bp_fc_low = 18.5e3 
pilot_bp_fc_high = 20.5e3
pilot_bp_taps = 101

audio_Fs = 48e3
mono_Fc = 16e3
mono_decim = 5
mono_taps = 101

# INIITIAL PLL STATES
pll_state_integrator = 0.0
pll_state_phaseEst = 0.0
pll_state_feedbackI = 1.0
pll_state_feedbackQ = 0.0
pll_state_trigOffset = 0
pll_state_lastNco = 1.0
# add other settings for audio, like filter taps, ...

# flag that keeps track if your code is running for
# in-lab (il_vs_th = 0) vs takehome (il_vs_th = 1)
il_vs_th = 0

if __name__ == "__main__":

	# read the raw IQ data from the recorded file
	# IQ data is assumed to be in 8-bits unsigned (and interleaved)
	in_fname = "../data/stereo_l0_r9.raw"

	raw_data = np.fromfile(in_fname, dtype='uint8')
	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
	# IQ data is normalized between -1 and +1 in 32-bit float format
	iq_data = (np.float32(raw_data) - 128.0)/128.0
	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

	# coefficients for the front-end low-pass filter
	rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))

	mono_coeff = signal.firwin(mono_taps, mono_Fc/((rf_Fs/rf_decim)/2), window=('hann'))

	stereo_coeff = bpFirwin((rf_Fs/rf_decim), 
						 	stereo_bp_fc_low, 
							stereo_bp_fc_high, 
							stereo_bp_taps)
	
	pilot_coeff = bpFirwin((rf_Fs/rf_decim), 
						 	pilot_bp_fc_low, 
							pilot_bp_fc_high, 
							pilot_bp_taps)
	
	# State variables
	state_i_lpf_100k = np.zeros(rf_taps-1)
	state_q_lpf_100k = np.zeros(rf_taps-1)
	state_stereo_bpf = np.zeros(stereo_bp_taps-1)
	state_pilot_bpf = np.zeros(pilot_bp_taps-1)
	state_phase = 0
	state_i_custom = np.float64(0.0)
	state_q_custom = np.float64(0.0)

	# set up the subfigures for plotting
	subfig_height = np.array([0.8, 2, 1.6])
	plt.rc('figure', figsize=(7.5, 7.5))	
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': subfig_height})
	fig.subplots_adjust(hspace = .6)

	# select a block_size that is a multiple of KB
	# and a multiple of decimation factors
	block_size = 1024 * rf_decim * mono_decim * 2
	block_count = 0

	# audio buffer that stores all the audio blocks
	mono_data = np.array([]) # used to concatenate filtered blocks (audio data)

	# state for the audio mono processing
	mono_lpf_state = np.zeros(mono_taps-1)
	mono_delay_state = np.zeros((state_stereo_bpf-1)/2)

	# if the number of samples in the last block is less than the block size
	# it is fine to ignore the last few samples from the raw IQ file
	while (block_count+1)*block_size < len(iq_data):

		# if you wish to have shorter runtimes while troubleshooting
		# you can control the above loop exit condition as you see fit
		print('Processing block ' + str(block_count))

		# filter to extract the FM channel (I samples are even, Q samples are odd)
		i_filt, state_i_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size:(block_count+1)*block_size:2],
				zi=state_i_lpf_100k)

		q_filt, state_q_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2],
				zi=state_q_lpf_100k)

		# downsample the I/Q data from the FM channel
		i_ds = i_filt[::rf_decim]
		q_ds = q_filt[::rf_decim]

		# FM demodulator
		fm_demod, state_phase = fmDemodArctan(i_ds, q_ds, state_phase)

		# Filter mono audio
		mono_filt, mono_lpf_state = signal.lfilter(mono_coeff, 1.0, fm_demod, zi=mono_lpf_state)

		pilot_filt, state_pilot_bpf = signal.lfilter(pilot_coeff, 1.0, fm_demod, zi=state_pilot_bpf)

		stereo_filt, state_stereo_bpf = signal.lfilter(stereo_coeff, 1.0, fm_demod, zi=state_stereo_bpf)

		ncoOut, pll_state_integrator, pll_state_phaseEst, pll_state_trigOffset, pll_state_lastNco = fmPll(
			pilot_filt, 
			19e3, 
			rf_Fs/rf_decim, 
			pll_state_integrator, 
			pll_state_phaseEst,
			pll_state_feedbackI,
			pll_state_feedbackQ,
			pll_state_trigOffset, 
			pll_state_lastNco, 
			ncoScale=2)

		# downsample audio data
		# to be updated by you during in-lab (same code for takehome)
		mono_block = mono_filt[::mono_decim]

		mono_block_delay, mono_delay_state = delayBlock(mono_block, mono_delay_state)

		# concatenate the most recently processed mono_block
		# to the previous blocks stored already in mono_data
		
		mono_data = np.concatenate((mono_data, mono_block))


		

		block_count += 1
	# loop end
	print('Finished processing all the blocks from the recorded I/Q samples')

	# write audio data to file
	out_fname = "../data/fmMonoBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((mono_data/2)*32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# uncomment assuming you wish to show some plots
	plt.show()

