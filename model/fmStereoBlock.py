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
	in_fname = "../data/samples9.raw"

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

	# state_phase = 0
	prev_i = 0.0
	prev_q = 0.0

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
	fmdemod_apf_state = np.zeros(int((stereo_lp_taps-1)/2))

	nco_debug = np.array([])
	pilot_debug = np.array([])


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
		# fm_demod, state_phase = fmDemodArctan(i_ds, q_ds, state_phase)
		fm_demod, prev_i, prev_q = custom_fm_demod(i_ds, q_ds, prev_i, prev_q)

		fm_demod_delayed, fmdemod_apf_state = delayBlock(fm_demod, fmdemod_apf_state)

		# Filter mono audio
		mono_filt, mono_lpf_state = signal.lfilter(mono_coeff, 1.0, fm_demod_delayed, zi=mono_lpf_state)
		mono_block = mono_filt[::mono_decim]

		pilot_filt, state_pilot_bpf = signal.lfilter(pilot_coeff, 1.0, fm_demod, zi=state_pilot_bpf)

		stereo_bpf_filtered, state_stereo_bpf = signal.lfilter(stereo_bpf_coeff, 1.0, fm_demod, zi=state_stereo_bpf)

		if block_count == 100:
			logVector(f"py_pll_state_in{block_count}", 
			[pll_state_integrator, 
			pll_state_phaseEst,
			pll_state_feedbackI,
			pll_state_feedbackQ,
			pll_state_trigOffset,
			pll_state_lastNco])

			logVector(f"py_pilot_filtered{block_count}", pilot_filt)



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

		if (block_count == 100 or block_count == 101):
			nco_debug = np.concatenate((nco_debug, ncoOut))
			pilot_debug = np.concatenate((pilot_debug, pilot_filt))
		elif block_count == 102:
			plt.plot(nco_debug)
			plt.plot(pilot_debug)
			plt.show()
		
		# import pdb; pdb.set_trace()
		
		if block_count == 100:
			logVector(f"py_pll_state_out{block_count}", 
			[pll_state_integrator, 
			pll_state_phaseEst,
			pll_state_feedbackI,
			pll_state_feedbackQ,
			pll_state_trigOffset,
			pll_state_lastNco])

			logVector(f"py_nco_out{block_count}", ncoOut)
		
		# if block_count == 100: import pdb; pdb.set_trace()
		
		# analog mixed
		stereo_mixed = np.zeros(len(stereo_bpf_filtered))
		for i in range(0,len(stereo_mixed)):
			stereo_mixed[i] = 2*ncoOut[i] * stereo_bpf_filtered[i]
		
		# if block_count == 10 or block_count == 1:

		# 	plt.plot(ncoOut)
		# 	fmPlotPSD(ax0, stereo_mixed, rf_Fs/rf_decim, subfig_height[0], "ncoOut")
		# 	plt.show()

		# low pass filter the stereo data
		stereo_filt_lpf, state_stereo_lpf = signal.lfilter(stereo_lpf_coeff, 1.0, stereo_mixed, zi=state_stereo_lpf)

		stereo_filt_lpf = stereo_filt_lpf

		# downsample stereo data
		stereo_block = stereo_filt_lpf[::stereo_decim]

		stereo_left_block = (mono_block + stereo_block)
		stereo_right_block = (mono_block - stereo_block)

		
		stereo_left = np.concatenate((stereo_left, (mono_block + stereo_block)))
		stereo_right = np.concatenate((stereo_right, (mono_block - stereo_block)))

		if block_count == 100 or block_count == 1:
			logVector(f"py_demodulated_samples{block_count}", fm_demod);	
			logVector(f"py_demodulated_samples_delayed{block_count}", fm_demod_delayed);	
			logVector(f"py_float_mono_data{block_count}", mono_block);
			logVector(f"py_pilot_filtered{block_count}", pilot_filt);
			logVector(f"py_nco_out{block_count}", ncoOut);
			logVector(f"py_stereo_mixed{block_count}", stereo_mixed);
			logVector(f"py_stereo_lpf_filtered{block_count}", stereo_block);
			logVector(f"py_float_stereo_left_data{block_count}", stereo_left_block);
			logVector(f"py_float_stereo_right_data{block_count}", stereo_right_block);
		


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
