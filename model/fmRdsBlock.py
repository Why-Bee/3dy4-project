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

rds_bp_fc_low = 54e3
rds_bp_fc_high = 60e3
rds_bp_taps = 101

rds_squared_bp_fc_low = 113.5e3
rds_squared_bp_fc_high = 114.5e3
rds_squared_bp_taps = 101

# state variables
state_i_lpf_100k = np.zeros(rf_taps-1)
state_q_lpf_100k = np.zeros(rf_taps-1)
rds_filt_state = np.zeros(rds_bp_taps-1)
rds_filt_carrier_state = np.zeros(rds_squared_bp_taps-1)
state_phase = 0
state_i_custom = np.float64(0.0)
state_q_custom = np.float64(0.0)

# INIITIAL PLL STATES
pll_state_integrator = 0.0
pll_state_phaseEst = 0.0
pll_state_feedbackI = 1.0
pll_state_feedbackQ = 0.0
pll_state_trigOffset = 0
pll_state_lastNco = 1.0

if __name__ == "__main__":

	# read the raw IQ data from the recorded file
	# IQ data is assumed to be in 8-bits unsigned (and interleaved)
	in_fname = "../data/samples9.raw"

	raw_data = np.fromfile(in_fname, dtype='uint8')
	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
	# IQ data is normalized between -1 and +1 in 32-bit float format
	iq_data = (np.float32(raw_data) - 128.0)/128.0
	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")
	
	# set up the subfigures for plotting
	subfig_height = np.array([0.8, 2, 1.6])
	plt.rc('figure', figsize=(7.5, 7.5))	
	fig, (ax0) = plt.subplots(nrows=1, gridspec_kw={'height_ratios': subfig_height})
	fig.subplots_adjust(hspace = .6)

	# coefficients for the front-end low-pass filter
	rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))
	
	rds_bpf_coeff = bpFirwin((rf_Fs/rf_decim), 
                              rds_bp_fc_low, 
                              rds_bp_fc_high, 
                              rds_bp_taps)
	
	rds_squared_bpf_coeff = bpFirwin((rf_Fs/rf_decim), 
                                      rds_squared_bp_fc_low, 
                                      rds_squared_bp_fc_high, 
                                      rds_squared_bp_taps)
	
	# State variables
	state_i_lpf_100k = np.zeros(rf_taps-1)
	state_q_lpf_100k = np.zeros(rf_taps-1)

	# select a block_size that is a multiple of KB
	# and a multiple of decimation factors
	block_size = 1024 * rf_decim * 10 * 2
	block_count = 0

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
		
        # filter rds data
		rds_filt, rds_filt_state = signal.lfilter(rds_bpf_coeff, 1.0, fm_demod, zi=rds_filt_state)
		
        # squaring non-linearity
		rds_filt_squared = rds_filt*rds_filt
		
        # extract the 114 kHz carrier
		rds_filt_carrier, rds_filt_carrier_state = signal.lfilter(rds_squared_bpf_coeff, 1.0, rds_filt_squared, zi=rds_filt_carrier_state)
		
		ncoOut, pll_state_integrator, pll_state_phaseEst, pll_state_trigOffset, pll_state_lastNco = fmPll(
			rds_filt_carrier, 
			114e3, 
			rf_Fs/rf_decim, 
			pll_state_integrator, 
			pll_state_phaseEst,
			pll_state_feedbackI,
			pll_state_feedbackQ,
			pll_state_trigOffset, 
			pll_state_lastNco, 
			ncoScale=0.5,
			normBandwidth=0.0025)
		
		if block_count == 20:
			plt.plot(ncoOut[:200])
			plt.plot(rds_filt_carrier[:200]*75)
			plt.show()
			
		block_count += 1

	# loop end
	print('Finished processing all the blocks from the recorded I/Q samples')

