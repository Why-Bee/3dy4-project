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

audio_Fs = 48e3
mono_Fc = 16e3
mono_decim = 5
mono_taps = 101

stereo_bp_fc_low = 22e3
stereo_bp_fc_high = 54e3
stereo_bp_taps = 101
stereo_decim = mono_decim

stereo_lp_fc = 38e3
stereo_lp_taps = 101

pilot_bp_fc_low = 18.5e3 
pilot_bp_fc_high = 20.5e3
pilot_bp_taps = 101

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

	stereo_bpf_coeff = bpFirwin((rf_Fs/rf_decim), 
						 	stereo_bp_fc_low, 
							stereo_bp_fc_high, 
							stereo_bp_taps)
	

	
	stereo_lpf_coeff = lpCoeff((rf_Fs/rf_decim), stereo_lp_fc, stereo_lp_taps) #signal.firwin(stereo_lp_taps, stereo_lp_fc/((rf_Fs/rf_decim)/2), window=('hann'))
	
	pilot_coeff = bpFirwin((rf_Fs/rf_decim), 
						 	pilot_bp_fc_low, 
							pilot_bp_fc_high, 
							pilot_bp_taps)
	
	# State variables
	state_i_lpf_100k = np.zeros(rf_taps-1)
	state_q_lpf_100k = np.zeros(rf_taps-1)
	state_stereo_bpf = np.zeros(stereo_bp_taps-1)
	state_stereo_lpf = np.zeros(stereo_lp_taps-1)
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
	stereo_left = np.array([]) # used to concatenate filtered blocks (audio data)
	stereo_right = np.array([]) # used to concatenate filtered blocks (audio data)
	stereo_left_right = np.array([]) # used to concatenate filtered blocks (audio data)

	# state for the audio mono processing
	mono_lpf_state = np.zeros(mono_taps-1)
	mono_apf_state = np.zeros(int((stereo_lp_taps-1)/2))

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

		stereo_filt_lpf_bpf, state_stereo_bpf = signal.lfilter(stereo_bpf_coeff, 1.0, fm_demod, zi=state_stereo_bpf)

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
			ncoScale=2,
			normBandwidth=0.01)
		
		# if block_count == 100: import pdb; pdb.set_trace()
		
		# analog mixer
		stereo_mixed = ncoOut[:-1] * stereo_filt_lpf_bpf
		
		# if block_count == 10:
		# 	plt.plot(ncoOut)
		# 	fmPlotPSD(ax0, stereo_mixed, rf_Fs/rf_decim, subfig_height[0], "ncoOut")
		# 	plt.show()

		# low pass filter the stereo data
		stereo_filt_lpf, state_stereo_lpf = signal.lfilter(stereo_lpf_coeff, 1.0, stereo_mixed, zi=state_stereo_lpf)

		# downsample stereo data
		stereo_block = 2*stereo_filt_lpf[::stereo_decim]

		# downsample mono data
		mono_filt_delayed, mono_apf_state = delayBlock(mono_filt, mono_apf_state)
		mono_block = mono_filt_delayed[::mono_decim]
		
		stereo_left = np.concatenate((stereo_left, (mono_block + stereo_block)))
		stereo_right = np.concatenate((stereo_right, (mono_block - stereo_block)))

		# import pdb; pdb.set_trace()

		block_count += 1
	# loop end
	print('Finished processing all the blocks from the recorded I/Q samples')

	# write audio data to file (assumes audio_data samples are -1 to +1)
	stereo_out_fname_left = "../data/fmStereoBlock_left.wav"
	stereo_out_fname_right = "../data/fmStereoBlock_right.wav"
	stereo_out_fname = '../data/fmStereoBlock_stereo.wav'
	wavfile.write(stereo_out_fname_left, int(audio_Fs), np.int16((stereo_left/2)*32767))
	wavfile.write(stereo_out_fname_right, int(audio_Fs), np.int16((stereo_right/2)*32767))
	c = np.column_stack((np.int16((stereo_left/2)*32767), np.int16((stereo_right/2)*32767)))
	print(c)
	wavfile.write(stereo_out_fname, int(audio_Fs), c)
	# during FM transmission audio samples in the mono channel will contain
	# the sum of the left and right audio channels; hence, we first
	# divide by two the audio sample value and then we rescale to fit
	# in the range offered by 16-bit signed int representation
	# print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")
